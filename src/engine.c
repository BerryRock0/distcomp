#include "engine.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Прямой проход llama-как-в-llama.cpp.
 *
 * Раскладка весов: тензор [ne0, ne1] лежит так, что элемент (i, j) стоит по
 * адресу i*ne1 + j, то есть y[j] = sum_i W[i][j] * x[i]. Это ровно то, что
 * делает llama.cpp, и именно поэтому готовые GGUF читаются без конвертации.
 *
 * Внутри malloc нет: все буферы берутся из арены, которую выделяет вызывающий.
 * На микроконтроллере это статический пул, на планировщике — malloc.
 */

#define ERR(...)                                                              \
    do {                                                                      \
        snprintf(err, errlen, __VA_ARGS__);                                   \
        return -1;                                                            \
    } while (0)

static float f16_at(const unsigned short *p)
{
    unsigned sign = (unsigned)(*p >> 15) & 1;
    unsigned exp = (unsigned)(*p >> 10) & 0x1F;
    unsigned man = (unsigned)*p & 0x3FF;
    uint32_t bits;
    float    out;

    if (exp == 0) {
        if (man == 0) {
            bits = sign << 31;
        } else {
            int e = -1;
            do { man <<= 1; e++; } while (!(man & 0x400));
            man &= 0x3FF;
            bits = (sign << 31) | ((uint32_t)(112 - e) << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    memcpy(&out, &bits, 4);
    return out;
}

static int read_shape(const gguf_file *g, const char *arch, llm_ctx *c, char *err, size_t errlen)
{
    unsigned long v;
    if (gguf_u32(g, &v, "block_count", arch) != 0) ERR("в модели нет %s.block_count", arch);
    c->n_layer = (int)v;
    if (gguf_u32(g, &v, "embedding_length", arch) != 0)
        ERR("в модели нет %s.embedding_length", arch);
    c->n_embd = (int)v;
    if (gguf_u32(g, &v, "feed_forward_length", arch) != 0)
        ERR("в модели нет %s.feed_forward_length", arch);
    c->n_ff = (int)v;
    if (gguf_u32(g, &v, "attention.head_count", arch) != 0)
        ERR("в модели нет %s.attention.head_count", arch);
    c->n_head = (int)v;
    if (gguf_u32(g, &v, "attention.head_count_kv", arch) != 0) c->n_head_kv = c->n_head;
    else                                                    c->n_head_kv = (int)v;

    unsigned long ctxlen = 512;
    gguf_u32(g, &ctxlen, "context_length", arch);
    c->n_ctx = (int)ctxlen;

    float eps = 1e-5f;
    if (gguf_f32(g, &eps, "attention.layer_norm_rms_epsilon", arch) != 0) {
        if (gguf_f32(g, &eps, "attention.layer_norm_epsilon", arch) != 0) eps = 1e-5f;
    }
    c->rms_eps = eps;

    float rope = 10000.0f;
    gguf_f32(g, &rope, "rope.freq_base", arch);
    c->rope_base = rope > 0 ? rope : 10000.0f;

    if (c->n_head <= 0 || c->n_embd % c->n_head)
        ERR("размерность %d не делится на число голов %d", c->n_embd, c->n_head);
    c->head_dim = c->n_embd / c->n_head;
    if (c->n_head_kv <= 0 || c->n_head % c->n_head_kv)
        ERR("число голов %d не делится на %d", c->n_head, c->n_head_kv);
    c->n_rep = c->n_head / c->n_head_kv;

    unsigned long rd = 0;
    if (gguf_u32(g, &rd, "rope.dimension_count", arch) == 0 && rd > 0) c->head_dim = (int)rd;

    const gguf_tensor *emb = gguf_tensor_find(g, "token_embd.weight");
    if (!emb) ERR("в модели нет token_embd.weight");
    if (emb->n_dims != 2) ERR("token_embd.weight: ожидалось 2 измерения");
    c->n_vocab = (int)emb->dims[1];
    if (c->n_ctx <= 0) c->n_ctx = 512;
    return 0;
}

size_t llm_ctx_floats(const gguf_file *g, int stage, int first, int last, int n_ctx)
{
    const char *arch = gguf_str(g, "general.architecture");
    if (!arch) return 0;
    llm_ctx c;
    char    err[192];
    memset(&c, 0, sizeof c);
    if (read_shape(g, arch, &c, err, sizeof err) != 0) return 0;
    if (n_ctx > 0) c.n_ctx = n_ctx;

    int nlayers = (stage == LLM_STAGE_LAYERS) ? last - first + 1 : 0;
    int wide = c.n_ff > c.n_embd ? c.n_ff : c.n_embd;

    size_t n = 0;
    n += (size_t)c.n_embd;                           /* xb */
    n += (size_t)wide;                               /* xb2: в него пишут и gate/up */
    n += (size_t)c.n_ff;                             /* xb3 */
    n += (size_t)c.n_embd;                           /* wn */
    n += (size_t)wide;                               /* xs */
    n += 2 * (size_t)c.n_head * c.head_dim;          /* q, att */
    n += 2 * (size_t)c.n_head_kv * c.head_dim;      /* k, v */
    n += (size_t)c.head_dim;                         /* qh */
    n += (size_t)c.n_ctx;                            /* scores */
    if (stage == LLM_STAGE_HEAD) n += (size_t)c.n_vocab;
    if (nlayers > 0) {
        size_t per = (size_t)c.n_ctx * c.n_head_kv * c.head_dim;
        n += 2 * (size_t)nlayers * per;
    }
    return n;
}

int llm_ctx_open(llm_ctx *c, const gguf_file *g, int stage, int first, int last,
                 int n_ctx, llm_layer *layer_store, float *arena, size_t arena_floats,
                 char *err, size_t errlen)
{
    memset(c, 0, sizeof *c);
    const char *arch = gguf_str(g, "general.architecture");
    if (!arch) ERR("в модели нет general.architecture");
    if (strcmp(arch, "llama") != 0)
        ERR("архитектура «%s» не поддерживается: движок умеет llama "
            "(LLaMA, Mistral, Qwen2)", arch);
    if (read_shape(g, arch, c, err, errlen) != 0) return -1;
    if (n_ctx > 0) c->n_ctx = n_ctx;
    if (stage == LLM_STAGE_LAYERS && (last < first || last >= c->n_layer))
        ERR("слои %d..%d вне модели: в ней %d слоёв", first, last, c->n_layer);

    c->stage = stage;
    c->first = first;
    c->last = last;

    if (stage == LLM_STAGE_EMBED) {
        c->w_emb.t = gguf_tensor_find(g, "token_embd.weight");
        if (!c->w_emb.t) ERR("в модели нет token_embd.weight");
    } else if (stage == LLM_STAGE_HEAD) {
        c->w_norm.t = gguf_tensor_find(g, "output_norm.weight");
        c->w_out.t = gguf_tensor_find(g, "output.weight");
        if (!c->w_norm.t) ERR("в модели нет output_norm.weight");
        if (!c->w_out.t) {                    /* веса могут быть связаны с эмбеддингом */
            c->w_out.t = gguf_tensor_find(g, "token_embd.weight");
            if (!c->w_out.t) ERR("в модели нет ни output.weight, ни token_embd.weight");
        }
    } else {
        int n = last - first + 1;
        if (n <= 0) ERR("диапазон слоёв пуст");
        /* слои — из переданного хранилища (на устройстве это статический массив,
         * malloc там недоступен) или через выделение на сервере */
        c->layers = layer_store ? layer_store : xcalloc((size_t)n, sizeof *c->layers);
        for (int i = 0; i < n; i++) {
            int       idx = first + i;
            llm_layer *l = &c->layers[i];
            llm_w     *w[] = { &l->attn_norm, &l->attn_q, &l->attn_k, &l->attn_v,
                               &l->attn_out,  &l->ffn_norm, &l->ffn_gate, &l->ffn_up,
                               &l->ffn_down };
            const char *sfx[] = { "attn_norm.weight", "attn_q.weight", "attn_k.weight",
                                  "attn_v.weight", "attn_output.weight", "ffn_norm.weight",
                                  "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight" };
            char name[128];
            for (size_t k = 0; k < sizeof w / sizeof w[0]; k++) {
                snprintf(name, sizeof name, "blk.%d.%s", idx, sfx[k]);
                w[k]->t = gguf_tensor_find(g, name);
                if (!w[k]->t) ERR("в модели нет тензор %s", name);
            }
        }
    }

    size_t need = llm_ctx_floats(g, stage, first, last, c->n_ctx);
    if (!arena || arena_floats < need)
        ERR("нужно %zu float под эту часть модели, дано %zu", need, arena_floats);

    int wide = c->n_ff > c->n_embd ? c->n_ff : c->n_embd;
    float *p = arena;
    c->xb = p;    p += c->n_embd;
    c->xb2 = p;   p += wide;
    c->xb3 = p;   p += c->n_ff;
    c->wn = p;    p += c->n_embd;
    c->xs = p;    p += wide;
    c->q = p;     p += (size_t)c->n_head * c->head_dim;
    c->att = p;   p += (size_t)c->n_head * c->head_dim;
    c->k = p;     p += (size_t)c->n_head_kv * c->head_dim;
    c->v = p;     p += (size_t)c->n_head_kv * c->head_dim;
    c->qh = p;    p += c->head_dim;
    c->scores = p; p += c->n_ctx;
    if (stage == LLM_STAGE_HEAD) { c->logits = p; p += (size_t)c->n_vocab; }
    if (stage == LLM_STAGE_LAYERS) {
        size_t per = (size_t)c->n_ctx * c->n_head_kv * c->head_dim;
        c->kc = p;  p += per * (size_t)(last - first + 1);
        c->vc = p;  p += per * (size_t)(last - first + 1);
    }
    return 0;
}

void llm_reset(llm_ctx *c)
{
    if (c) c->pos = 0;
}

/* ---------------------------------------------------------------- слои --- */

/* RMSNorm: x / sqrt(mean(x^2) + eps) * w */
static void rms_norm(float *dst, const float *x, const float *w, int n, float eps)
{
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) dst[i] = x[i] * scale * w[i];
}

/* y[j] = sum_i W[i*ne1 + j] * x[i]. F16 разворачивается построчно: так
 * внутренний цикл остаётся с плавающей арифметикой, а вызовов на каждый вес
 * не будет. */
static void matvec(float *y, const gguf_tensor *w, const float *x, int n_in, int n_out,
                   float *scratch)
{
    const unsigned char *raw = w->data;
    int                 is_f16 = w->type == 1;

    for (int j = 0; j < n_out; j++) y[j] = 0.0f;
    for (int i = 0; i < n_in; i++) {
        float xi = x[i];
        const unsigned char *row = raw + (size_t)i * (size_t)n_out * (is_f16 ? 2u : 4u);
        const float          *r;
        if (is_f16) {
            const unsigned short *h = (const unsigned short *)(const void *)row;
            for (int j = 0; j < n_out; j++) scratch[j] = f16_at(h + j);
            r = scratch;
        } else {
            r = (const float *)(const void *)row;
        }
        for (int j = 0; j < n_out; j++) y[j] += r[j] * xi;
    }
}

/* разворачивает один ряд тензора в float */
static void row_to_float(const gguf_tensor *t, float *dst, size_t off, size_t n)
{
    const unsigned char *p = t->data + off * (t->type == 1 ? 2u : 4u);
    if (t->type == 0) {
        memcpy(dst, p, sizeof(float) * n);
    } else {
        const unsigned short *h = (const unsigned short *)(const void *)p;
        for (size_t i = 0; i < n; i++) dst[i] = f16_at(h + i);
    }
}

/* RoPE: пары (x, y) поворачиваются на угол pos * base^(-2i/d) */
static void rope(float *v, int n, int pos, float base)
{
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        float freq = powf(base, -2.0f * (float)i / (float)n);
        float ang = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float a = v[i], b = v[half + i];
        v[i] = a * c - b * s;
        v[half + i] = b * c + a * s;
    }
}

static void forward_layer(llm_ctx *c, llm_layer *l, int li, float *x)
{
    int ne = c->n_embd, hd = c->head_dim;
    int nh = c->n_head, nkv = c->n_head_kv;

    row_to_float(l->attn_norm.t, c->wn, 0, (size_t)ne);
    rms_norm(c->xb, x, c->wn, ne, c->rms_eps);

    matvec(c->q, l->attn_q.t, c->xb, ne, nh * hd, c->xs);
    matvec(c->k, l->attn_k.t, c->xb, ne, nkv * hd, c->xs);
    matvec(c->v, l->attn_v.t, c->xb, ne, nkv * hd, c->xs);

    size_t  kv_dim = (size_t)nkv * hd;
    size_t  plane = (size_t)c->n_ctx * kv_dim;
    float  *kbase = c->kc + (size_t)li * plane;
    float  *vbase = c->vc + (size_t)li * plane;
    float  *krow = kbase + (size_t)c->pos * kv_dim;
    float  *vrow = vbase + (size_t)c->pos * kv_dim;

    /* своя позиция в кэш: сначала без поворота, потом RoPE на месте */
    for (int j = 0; j < nkv; j++) {
        memcpy(krow + (size_t)j * hd, c->k + (size_t)j * hd, sizeof(float) * (size_t)hd);
        memcpy(vrow + (size_t)j * hd, c->v + (size_t)j * hd, sizeof(float) * (size_t)hd);
        rope(krow + (size_t)j * hd, hd, c->pos, c->rope_base);
    }

    int pos = c->pos + 1;
    for (int g = 0; g < nh; g++) {
        int kvg = g / c->n_rep;

        memcpy(c->qh, c->q + (size_t)g * hd, sizeof(float) * (size_t)hd);
        rope(c->qh, hd, c->pos, c->rope_base);

        float mx = -3.4e38f, sum = 0;
        for (int t = 0; t < pos; t++) {
            const float *kk = kbase + (size_t)t * kv_dim + (size_t)kvg * hd;
            float dot = 0;
            for (int j = 0; j < hd; j++) dot += c->qh[j] * kk[j];
            dot *= 1.0f / sqrtf((float)hd);
            c->scores[t] = dot;
            if (dot > mx) mx = dot;
        }
        for (int t = 0; t < pos; t++) {
            float e = expf(c->scores[t] - mx);
            c->scores[t] = e;
            sum += e;
        }
        float *acc = c->att + (size_t)g * hd;
        for (int j = 0; j < hd; j++) acc[j] = 0.0f;
        float inv = sum > 0 ? 1.0f / sum : 0.0f;
        for (int t = 0; t < pos; t++) {
            const float *vv = vbase + (size_t)t * kv_dim + (size_t)kvg * hd;
            float wgt = c->scores[t] * inv;
            for (int j = 0; j < hd; j++) acc[j] += vv[j] * wgt;
        }
    }

    matvec(c->xb2, l->attn_out.t, c->att, nh * hd, ne, c->xs);
    for (int i = 0; i < ne; i++) x[i] += c->xb2[i];

    /* SwiGLU: silu(gate) * up -> down */
    row_to_float(l->ffn_norm.t, c->wn, 0, (size_t)ne);
    rms_norm(c->xb, x, c->wn, ne, c->rms_eps);
    matvec(c->xb2, l->ffn_gate.t, c->xb, ne, c->n_ff, c->xs);
    matvec(c->xb3, l->ffn_up.t, c->xb, ne, c->n_ff, c->xs);
    for (int j = 0; j < c->n_ff; j++) {
        float g = c->xb2[j];
        c->xb2[j] = (g / (1.0f + expf(-g))) * c->xb3[j];
    }
    matvec(c->xb, l->ffn_down.t, c->xb2, c->n_ff, ne, c->xs);
    for (int i = 0; i < ne; i++) x[i] += c->xb[i];
}

int llm_forward(llm_ctx *c, int token, float *out)
{
    if (!c || !out) return -1;

    if (c->stage == LLM_STAGE_EMBED) {
        /* Таблица эмбеддингов [n_embd, n_vocab] лежит так, что элемент (i, j)
         * стоит по адресу i*n_vocab + j. Вектор токена — это столбец j, то есть
         * числа идут через шаг n_vocab, а не подряд. */
        const gguf_tensor *t = c->w_emb.t;
        int n_rows = t->n_dims == 2 ? (int)t->dims[1] : c->n_vocab;
        int ne = c->n_embd;
        if (token < 0 || token >= n_rows) return -1;
        size_t step = (t->n_dims == 2) ? (size_t)t->dims[1] : (size_t)n_rows;
        for (int i = 0; i < ne; i++)
            row_to_float(t, out + i, (size_t)i * step + (size_t)token, 1);
        return 0;
    }

    if (c->stage == LLM_STAGE_LAYERS) {
        return llm_step_layers(c, 0, out);
    }

    row_to_float(c->w_norm.t, c->wn, 0, (size_t)c->n_embd);
    rms_norm(c->xb, out, c->wn, c->n_embd, c->rms_eps);
    matvec(c->logits, c->w_out.t, c->xb, c->n_embd, c->n_vocab, c->xs);
    memcpy(out, c->xb, sizeof(float) * (size_t)c->n_embd);
    out[c->n_embd] = (float)llm_pick(c);
    return 0;
}

int llm_layer_once(llm_ctx *c, int layer_idx, float *out)
{
    if (!c || c->stage != LLM_STAGE_LAYERS || !out) return -1;
    if (layer_idx < 0 || layer_idx > c->last - c->first) return -1;
    forward_layer(c, &c->layers[layer_idx], layer_idx, out);
    return 0;
}

int llm_step_layers(llm_ctx *c, int n, float *out)
{
    if (!c || c->stage != LLM_STAGE_LAYERS || !out) return -1;
    if (n <= 0) n = c->last - c->first + 1;
    if (c->pos >= c->n_ctx) return -1;        /* контекст исчерпан */
    for (int i = 0; i < n; i++) forward_layer(c, &c->layers[i], i, out);
    c->pos++;
    return 0;
}

int llm_pick(const llm_ctx *c)
{
    if (!c || !c->logits) return 0;
    int   best = 0;
    float bv = c->logits[0];
    for (int i = 1; i < c->n_vocab; i++) {
        if (c->logits[i] > bv) { bv = c->logits[i]; best = i; }
    }
    return best;
}

#undef ERR