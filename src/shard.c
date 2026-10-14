#include "shard.h"
#include "engine.h"
#include "gguf.h"
#include "json.h"
#include "sched.h"
#include "tok.h"
#include "udp.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

/*
 * Сторона планировщика: LLM, разложенная по устройствам.
 *
 * Модель читается из GGUF — того самого формата, в котором её отдают
 * llama.cpp, Ollama и vLLM. Дальше модель режется на стадии (вход, слои,
 * выход), стадии раскладываются по узлам роли shard пропорционально их свободной
 * памяти, а по сети идёт не модель, а активация: токен входит в узел входа,
 * состояние перетекает с узла на узел по порядку слоёв, и последний узел
 * выбирает следующий токен.
 *
 * Численная сверка: планировщик прогоняет ту же модель целиком у себя и
 * сравнивает с тем, что вернул конвейер. Если токены разойдутся — раскладка
 * или пересылка где-то испортились, и это видно сразу.
 *
 * Что поддержано честно и что нет:
 *   - архитектура llama (LLaMA, Mistral, Qwen2, родственники);
 *   - веса F32 и F16, без квантованных форматов;
 *   - жадная генерация (argmax) — температура и top-k не реализованы.
 */

/* ---------------------------------------------------------------- модель --- */

typedef struct {
    unsigned char *raw;
    size_t         size;
    gguf_file      g;
    llm_tok        tok;

    /* геометрия для раскладки */
    int    n_layer, n_embd, n_head, n_head_kv, n_ff, head_dim, n_vocab, n_ctx;
    float  rms_eps, rope_base;
    const char *arch;
} llm_model;

static void llm_free(llm_model *m)
{
    if (!m) return;
    llm_tok_close(&m->tok);
    gguf_close(&m->g);
    free(m->raw);
    memset(m, 0, sizeof *m);
}

static int llm_open_file(llm_model *m, const char *path, char *err, size_t errlen)
{
    memset(m, 0, sizeof *m);
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errlen, "не удалось открыть %s", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        snprintf(err, errlen, "файл %s пуст", path);
        fclose(f);
        return -1;
    }
    m->raw = xmalloc((size_t)sz);
    if (fread(m->raw, 1, (size_t)sz, f) != (size_t)sz) {
        snprintf(err, errlen, "файл %s прочитан не полностью", path);
        fclose(f);
        return -1;
    }
    fclose(f);
    m->size = (size_t)sz;

    char e[512] = {0};
    if (gguf_open(&m->g, m->raw, m->size, e, sizeof e) != 0) {
        snprintf(err, errlen, "GGUF: %s", e);
        llm_free(m);
        return -1;
    }
    m->arch = gguf_str(&m->g, "general.architecture");
    if (!m->arch) {
        snprintf(err, errlen, "в модели нет general.architecture");
        llm_free(m);
        return -1;
    }

    unsigned long v;
    if (gguf_u32(&m->g, &v, "block_count", m->arch) == 0) m->n_layer = (int)v;
    if (gguf_u32(&m->g, &v, "embedding_length", m->arch) == 0) m->n_embd = (int)v;
    if (gguf_u32(&m->g, &v, "feed_forward_length", m->arch) == 0) m->n_ff = (int)v;
    if (gguf_u32(&m->g, &v, "attention.head_count", m->arch) == 0) m->n_head = (int)v;
    if (gguf_u32(&m->g, &v, "attention.head_count_kv", m->arch) == 0) m->n_head_kv = (int)v;
    if (gguf_u32(&m->g, &v, "context_length", m->arch) == 0) m->n_ctx = (int)v;
    if (gguf_f32(&m->g, &m->rms_eps, "attention.layer_norm_rms_epsilon", m->arch) != 0)
        m->rms_eps = 1e-5f;
    if (gguf_f32(&m->g, &m->rope_base, "rope.freq_base", m->arch) != 0)
        m->rope_base = 10000.0f;
    if (gguf_u32(&m->g, &v, "rope.dimension_count", m->arch) == 0 && v > 0)
        m->head_dim = (int)v;
    else if (m->n_head > 0)
        m->head_dim = m->n_embd / m->n_head;

    const gguf_tensor *emb = gguf_tensor_find(&m->g, "token_embd.weight");
    if (emb && emb->n_dims == 2) m->n_vocab = (int)emb->dims[1];

    /* контекст узла: меньше просит — меньше памяти под KV-кэш */
    if (m->n_ctx <= 0 || m->n_ctx > 512) m->n_ctx = 512;

    if (llm_tok_open(&m->tok, &m->g, e, sizeof e) != 0) {
        snprintf(err, errlen, "токенизатор: %s", e);
        llm_free(m);
        return -1;
    }
    return 0;
}

/* Тензоры стадии: имена и суммарный размер в байтах. */
typedef struct {
    char        name[64];   /* имя копируем: строка собирается в локальном буфере */
    size_t      nbytes;
    const gguf_tensor *t;
} stage_tensor;

static int stage_tensors(const llm_model *m, int stage, int first, int last,
                        stage_tensor *out, int max, size_t *total_bytes)
{
    *total_bytes = 0;
    int n = 0;
    const char *name;

#define ADD(nn)                                                            \
    do {                                                                   \
        const gguf_tensor *t = gguf_tensor_find(&m->g, nn);                \
        if (!t) {                                                          \
            snprintf(err_tmp, sizeof err_tmp, "в модели нет тензор %s", nn); \
            return -1;                                                     \
        }                                                                  \
        if (n >= max) return -1;                                           \
        snprintf(out[n].name, sizeof out[n].name, "%s", nn);               \
        out[n].nbytes = t->nbytes;                                         \
        out[n].t = t;                                                      \
        *total_bytes += t->nbytes;                                         \
        n++;                                                               \
    } while (0)

    char err_tmp[128];
    char nm[128];

    if (stage == LLM_STAGE_EMBED) {
        ADD("token_embd.weight");
    } else if (stage == LLM_STAGE_HEAD) {
        ADD("output_norm.weight");
        ADD("output.weight");
        if (!gguf_tensor_find(&m->g, "output.weight")) {
            /* веса могут быть связаны с эмбеддингом: output уже положен как
             * token_embd в стадию входа, а здесь нужен тот же тензор */
            out[n - 1].name[0] = '\0';
            snprintf(out[n - 1].name, sizeof out[n - 1].name, "%s", "token_embd.weight");
        }
    } else {
        const char *sfx[] = { "attn_norm.weight", "attn_q.weight", "attn_k.weight",
                              "attn_v.weight", "attn_output.weight", "ffn_norm.weight",
                              "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight" };
        for (int i = first; i <= last; i++) {
            for (size_t k = 0; k < sizeof sfx / sizeof sfx[0]; k++) {
                snprintf(nm, sizeof nm, "blk.%d.%s", i, sfx[k]);
                ADD(nm);
            }
        }
    }
    return n;
#undef ADD
}

/* Ограничение на кусок данных: base64 распухает в 4/3, плюс заголовок
 * команды и запас на ответ. */
static long chunk_budget(const dc_endpoint *ep)
{
    long body = ep->cap.max_body > 96 ? ep->cap.max_body - 96 : 192;
    long raw = body * 3 / 4;
    return raw < 3 ? 3 : raw;
}

/* ------------------------------------------------------------------ узлы --- */

typedef struct {
    dc_endpoint *ep;
    long         cap;      /* free_mem: сколько байт под веса отдаёт */
    long         used;
    int          stage, first, last;   /* какая стадия лежит */
} llm_node;

static int collect_nodes(dc_config *cfg, llm_node *out, int max, char *err, size_t errlen)
{
    dc_role *r = conf_role(cfg, ROLE_SHARD, 0);
    if (!r || r->nep == 0) {
        snprintf(err, errlen,
                 "нет ни одного узла роли %s: добавьте секцию [role %s] daemon = host:port "
                 "для устройств, на которых будет лежать модель",
                 ROLE_SHARD, ROLE_SHARD);
        return -1;
    }
    int n = 0;
    for (int i = 0; i < r->nep && n < max; i++) {
        dc_endpoint *e = &r->eps[i];
        if (!e->cap.known) {
            snprintf(err, errlen, "узел %s не ответил — неизвестно, сколько памяти он отдаёт",
                     e->url);
            return -1;
        }
        if (e->cap.free_mem <= 0) {
            snprintf(err, errlen,
                     "узел %s не сообщил свободной памяти (free_mem=0): запустите его "
                     "с --free-mem, если память под модель надо ограничить явно",
                     e->url);
            return -1;
        }
        out[n].ep = e;
        out[n].cap = e->cap.free_mem;
        out[n].used = 0;
        out[n].stage = -1;
        out[n].first = out[n].last = -1;
        n++;
    }
    return n;
}

/* Сколько байт памяти узла требует стадия: веса плюс арена под активации. */
static long stage_cost(const llm_model *m, int stage, int first, int last)
{
    stage_tensor st[128];
    size_t total = 0;
    char err[128];
    int n = stage_tensors(m, stage, first, last, st, 128, &total);
    if (n < 0) return -1;
    /* арена: рабочие буферы и KV-кэш; считаем через тот же код, что у узла */
    size_t floats = llm_ctx_floats(&m->g, stage, first, last, m->n_ctx);
    long arena = (long)floats * 4;
    /* арена есть и у стадии входа/выхода (там KV-кэша нет, но буферы есть) */
    return (long)total + arena;
}

/* Стадии раскладываем так, чтобы каждый узел получил связный кусок конвейера:
 * по сети ходит состояние между соседними стадиями, и рванная раскладка
 * гоняла бы его по кругу. */
static int split_stages(const llm_model *m, llm_node *nodes, int nn, int *owner,
                        int *nstages_out, char *err, size_t errlen)
{
    if (nn < 3) {
        snprintf(err, errlen, "нужно минимум 3 узла роли %s (вход, слои, выход), а их %d",
                 ROLE_SHARD, nn);
        return -1;
    }

    int ns = 0;

    /* Стадия 0 — вход. Узел держит ровно один контекст движка, поэтому вход,
     * слои и выход обязаны жить на разных узлах. */
    long need = stage_cost(m, LLM_STAGE_EMBED, 0, 0);
    if (need < 0) { snprintf(err, errlen, "во входе модели не хватает тензоров"); return -1; }
    if (need > nodes[0].cap) {
        snprintf(err, errlen,
                 "вход (эмбеддинг) требует %ld байт, а узел %s отдаёт %ld: увеличьте --free-mem",
                 need, nodes[0].ep->url, nodes[0].cap);
        return -1;
    }
    owner[ns] = 0;
    nodes[0].stage = LLM_STAGE_EMBED;
    nodes[0].first = nodes[0].last = 0;
    nodes[0].used = need;
    ns++;

    /* Средние узлы — связные диапазоны слоёв 0..n_layer-1. Делим поровну,
     * ужимая диапазон под память узла; по одному слою резервируем на каждый
     * оставшийся узел, чтобы ни один слой не потерялся. */
    int l = 0;
    int mid_first = 1, mid_last = nn - 2;
    for (int k = mid_first; k <= mid_last && l < m->n_layer; k++) {
        int rem_layers = m->n_layer - l;
        int rem_nodes = mid_last - k + 1;
        int give = (rem_layers + rem_nodes - 1) / rem_nodes;   /* ceil(rem_layers/rem_nodes) */
        int max_give = rem_layers - (rem_nodes - 1);           /* остальным — хотя бы по слою */
        if (give > max_give) give = max_give;
        if (give < 1) give = 1;
        while (give >= 1 && stage_cost(m, LLM_STAGE_LAYERS, l, l + give - 1) > nodes[k].cap)
            give--;
        if (give < 1) {
            snprintf(err, errlen,
                     "узел %s не вмещает ни одного слоя (нужно %ld байт): увеличьте --free-mem",
                     nodes[k].ep->url, stage_cost(m, LLM_STAGE_LAYERS, l, l));
            return -1;
        }
        owner[ns] = k;
        nodes[k].stage = LLM_STAGE_LAYERS;
        nodes[k].first = l;
        nodes[k].last = l + give - 1;
        nodes[k].used = stage_cost(m, LLM_STAGE_LAYERS, nodes[k].first, nodes[k].last);
        l += give;
        ns++;
    }
    if (l < m->n_layer) {
        snprintf(err, errlen,
                 "не хватает узлов: размещено %d слоёв из %d — добавьте устройства",
                 l, m->n_layer);
        return -1;
    }

    /* Последняя стадия — выход. */
    need = stage_cost(m, LLM_STAGE_HEAD, 0, 0);
    if (need < 0) { snprintf(err, errlen, "в выходе модели не хватает тензоров"); return -1; }
    if (need > nodes[nn - 1].cap) {
        snprintf(err, errlen,
                 "выход (норма + голова) требует %ld байт, а узел %s отдаёт %ld: увеличьте --free-mem",
                 need, nodes[nn - 1].ep->url, nodes[nn - 1].cap);
        return -1;
    }
    owner[ns] = nn - 1;
    nodes[nn - 1].stage = LLM_STAGE_HEAD;
    nodes[nn - 1].first = nodes[nn - 1].last = 0;
    nodes[nn - 1].used = need;
    ns++;

    if (nstages_out) *nstages_out = ns;
    return 0;
}

static void print_layout(const llm_model *m, const llm_node *nodes, int nn, const int *owner,
                         int nstages)
{
    printf("\nРаскладка модели по устройствам:\n\n");
    printf("  стадия                требует    узел\n");
    for (int s = 0; s < nstages; s++) {
        const llm_node *nd = &nodes[owner[s]];
        char label[64];
        if (nd->stage == LLM_STAGE_EMBED) snprintf(label, sizeof label, "вход (эмбеддинг)");
        else if (nd->stage == LLM_STAGE_HEAD) snprintf(label, sizeof label, "выход (голова)");
        else if (nd->first == nd->last) snprintf(label, sizeof label, "слой %d", nd->first);
        else snprintf(label, sizeof label, "слои %d..%d", nd->first, nd->last);
        printf("  %-21s %-10ld %s\n", label, stage_cost(m, nd->stage, nd->first, nd->last),
               nd->ep->url);
    }
    printf("\n");
    for (int k = 0; k < nn; k++)
        if (nodes[k].stage >= 0)
            printf("  %-28s занято %ld из %ld байт\n", nodes[k].ep->url, nodes[k].used,
                   nodes[k].cap);
    printf("\n");
}

/* ------------------------------------------------------------ локальный --- */

/* Прогон всей модели у себя — эталон для сверки с конвейером. */
typedef struct {
    llm_ctx emb, lay, head;
    float *a_emb, *a_lay, *a_head, *state;
} local_run;

static int local_open(local_run *L, const llm_model *m, char *err, size_t errlen)
{
    memset(L, 0, sizeof *L);
    size_t ne = llm_ctx_floats(&m->g, LLM_STAGE_EMBED, 0, 0, 0);
    size_t nl = llm_ctx_floats(&m->g, LLM_STAGE_LAYERS, 0, m->n_layer - 1, m->n_ctx);
    size_t nh = llm_ctx_floats(&m->g, LLM_STAGE_HEAD, 0, 0, 0);
    L->a_emb = xcalloc(ne ? ne : 1, sizeof(float));
    L->a_lay = xcalloc(nl ? nl : 1, sizeof(float));
    L->a_head = xcalloc(nh ? nh : 1, sizeof(float));
    if (llm_ctx_open(&L->emb, &m->g, LLM_STAGE_EMBED, 0, 0, 0, NULL, L->a_emb, ne, err,
                     errlen) != 0 ||
        llm_ctx_open(&L->lay, &m->g, LLM_STAGE_LAYERS, 0, m->n_layer - 1, m->n_ctx, NULL,
                     L->a_lay, nl, err, errlen) != 0 ||
        llm_ctx_open(&L->head, &m->g, LLM_STAGE_HEAD, 0, 0, 0, NULL, L->a_head, nh, err,
                     errlen) != 0)
        return -1;
    L->state = xcalloc((size_t)m->n_embd + 8, sizeof(float));
    return 0;
}

static void local_free(local_run *L)
{
    free(L->a_emb); free(L->a_lay); free(L->a_head); free(L->state);
    memset(L, 0, sizeof *L);
}

/* Один шаг локального прогона: возвращает выбранный токен. */
/* Один шаг эталона. KV-кэш НЕ сбрасываем: контекст накапливается так же, как у
 * узлов конвейера, иначе эталон считал бы каждый токен в отрыве от истории. */
static int local_forward(local_run *L, int token)
{
    if (llm_forward(&L->emb, token, L->state) != 0) return -1;
    if (llm_forward(&L->lay, token, L->state) != 0) return -1;
    if (llm_forward(&L->head, token, L->state) != 0) return -1;
    return llm_pick(&L->head);
}

/* ------------------------------------------------------------ конвейер --- */

/* Стадии по порядку: вход, слои, выход. Каждая — на своём узле. */
typedef struct {
    int sock;
    const llm_model *m;
    const llm_node *nodes;
    const int *owner;
    int nstages;
    float *state;      /* текущее состояние, n_embd */
    int    n_embd;
} pipeline;

/* Отправляет payload узлу, разбирает JSON-ответ и отдаёт поле output.
 * Ошибка транспорта, неразобранный JSON или ok=false — это -1 с текстом в err. */
static int pipe_send(int sock, const dc_endpoint *ep, const char *payload, char **out_body,
                     char *err, size_t errlen)
{
    char host[128];
    int  port;
    if (udp_parse_endpoint(ep->url, host, sizeof host, &port, err, errlen) != 0) return -1;

    /* Узел принимает только JSON-запрос с полем role; команда конвейера едет
     * в поле payload. Собираем конверт и экранируем payload как строку. */
    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\"role\":\"");
    sb_add(&b, ROLE_SHARD);
    sb_add(&b, "\",\"title\":\"\",\"payload\":");
    sb_add_json_string(&b, payload);
    sb_add(&b, "}");

    char *resp = NULL;
    int   rc   = udp_request_raw(sock, host, port, b.p, strlen(b.p), 5000, &resp, err, errlen);
    sb_free(&b);
    if (rc != 0) return -1;

    char    perr[256] = {0};
    js_doc *doc = js_parse(resp ? resp : "", perr, sizeof perr);
    free(resp);
    if (!doc) {
        snprintf(err, errlen, "ответ узла %s не JSON: %s", ep->url, perr);
        return -1;
    }
    const js_value *root = js_root(doc);
    const js_value *okv  = js_get(root, "ok");
    int             ok   = (okv && okv->type == JS_BOOL) ? okv->bval : 1;
    const char     *out  = js_get_str(root, "output");
    const char     *errm = js_get_str(root, "error");
    if (!ok) {
        snprintf(err, errlen, "узел %s отказал: %s", ep->url, errm && *errm ? errm : "ok=false");
        js_doc_free(doc);
        return -1;
    }
    if (out_body) *out_body = xstrdup(out ? out : "");
    js_doc_free(doc);
    return 0;
}

/* Заливает стадию на узел: тензорами, потом shape. */
static int pipe_deploy_stage(pipeline *P, int s)
{
    llm_model *m = (llm_model *)P->m;
    const llm_node *nd = &P->nodes[P->owner[s]];
    stage_tensor st[128];
    size_t total;
    char err[512];
    int n = stage_tensors(m, nd->stage, nd->first, nd->last, st, 128, &total);
    if (n < 0) return -1;

    /* сначала освобождаем узел: вторая модель не влезет поверх первой */
    char *out = NULL;
    if (pipe_send(P->sock, nd->ep, "free", &out, err, sizeof err) != 0) {
        fprintf(stderr, "узел %s не освободился: %s\n", nd->ep->url, err);
        return -1;
    }
    free(out);

    for (int i = 0; i < n; i++) {
        long budget = chunk_budget(nd->ep);
        long off = 0;
        while (off < (long)st[i].nbytes) {
            long take = (long)st[i].nbytes - off;
            if (take > budget) take = budget;
            long chunk = take / 3 * 3;
            if (chunk <= 0) chunk = take;

            strbuf b;
            sb_init(&b);
            sb_printf(&b, "wt|%s|%u|", st[i].name, st[i].t->n_dims);
            for (unsigned d = 0; d < st[i].t->n_dims; d++)
                sb_printf(&b, "%s%lu", d ? "," : "", st[i].t->dims[d]);
            sb_printf(&b, "|%u|%ld|%ld|", st[i].t->type, off, (long)st[i].t->nbytes);
            char *b64 = xmalloc(b64_encode_size((size_t)chunk) + 1);
            b64_encode(st[i].t->data + off, (size_t)chunk, b64, b64_encode_size((size_t)chunk) + 1);
            sb_add(&b, b64);
            free(b64);

            out = NULL;
            if (pipe_send(P->sock, nd->ep, b.p, &out, err, sizeof err) != 0) {
                fprintf(stderr, "тензор %s не ушёл на %s: %s\n", st[i].name, nd->ep->url, err);
                sb_free(&b);
                return -1;
            }
            free(out);
            sb_free(&b);
            off += chunk;
        }
    }

    /* shape: геометрия и стадия */
    strbuf b;
    sb_init(&b);
    sb_printf(&b, "shape|%d|%d|%d|%d|%d|%d|%d|%d|%d|%d|%.8g|%.8g|%d",
              nd->stage, nd->first, nd->last, m->n_embd, m->n_head, m->n_head_kv, m->n_ff,
              m->head_dim, m->n_vocab, m->n_layer, (double)m->rms_eps, (double)m->rope_base,
              m->n_ctx);
    out = NULL;
    if (pipe_send(P->sock, nd->ep, b.p, &out, err, sizeof err) != 0) {
        fprintf(stderr, "shape не ушёл на %s: %s\n", nd->ep->url, err);
        sb_free(&b);
        return -1;
    }
    sb_free(&b);
    /* ответ содержит описание стадии; ошибка видна по префиксу ОШИБКА */
    if (strncmp(out ? out : "", "ОШИБКА", sizeof("ОШИБКА") - 1) == 0) {
        fprintf(stderr, "узел %s отказал: %s\n", nd->ep->url, out);
        free(out);
        return -1;
    }
    free(out);
    return 0;
}

/* Один шаг по конвейеру: токен -> выбранный следующий токен. */
static int pipe_step(pipeline *P, int token, char *err, size_t errlen)
{
    int nstages = P->nstages;
    for (int s = 0; s < nstages; s++) {
        const llm_node *nd = &P->nodes[P->owner[s]];

        strbuf b;
        sb_init(&b);
        sb_printf(&b, "gen|%d", token);
        if (s > 0) {
            /* прикладываем состояние от предыдущей стадии: по сети идёт активация */
            size_t nb  = (size_t)P->n_embd * sizeof(float);
            char  *b64 = xmalloc(b64_encode_size(nb) + 1);
            b64_encode(P->state, nb, b64, b64_encode_size(nb) + 1);
            sb_printf(&b, "|%s", b64);
            free(b64);
        }
        char *out = NULL;
        if (pipe_send(P->sock, nd->ep, b.p, &out, err, errlen) != 0) {
            sb_free(&b);
            return -1;
        }
        /* ответ стадии выхода — выбранный токен. Префикс «токен » кириллический:
         * в UTF-8 это 11 байт, поэтому длину берём из sizeof, а не «на глаз». */
        if (s == nstages - 1 && strncmp(out ? out : "", "токен ", sizeof("токен ") - 1) == 0) {
            int tok = atoi((out ? out : "") + (sizeof("токен ") - 1));
            free(out);
            sb_free(&b);
            return tok;
        }
        free(out);
        sb_free(&b);

        /* забираем состояние частями */
        long got = 0;
        long want = (long)P->n_embd * 4;
        while (got < want) {
            long n = want - got;
            if (n > 4096) n = 4096;
            sb_init(&b);
            sb_printf(&b, "part|%ld|%ld", got, n);
            out = NULL;
            if (pipe_send(P->sock, nd->ep, b.p, &out, err, errlen) != 0) {
                sb_free(&b);
                return -1;
            }
            size_t dec = 0;
            if (b64_decode(out ? out : "", (char *)P->state + got, (size_t)n, &dec) != 0) {
                snprintf(err, errlen, "не разобран кусок состояния от %s", nd->ep->url);
                free(out);
                sb_free(&b);
                return -1;
            }
            got += (long)dec;
            free(out);
            sb_free(&b);
        }
    }
    return -1;
}

/* ------------------------------------------------------------- команды --- */

static void usage(void)
{
    printf(
        "dcctl llm — LLM, разложенная по устройствам\n"
        "\n"
        "  llm info ФАЙЛ                            раскладка по узлам\n"
        "  llm deploy ФАЙЛ                          раздать веса по узлам\n"
        "  llm prompt ФАЙЛ \"текст\" [токенов]       генерация с проверкой\n"
        "\n"
        "Модель читается из GGUF (F32/F16, архитектура llama). Узлы берутся из\n"
        "секции [role %s]: стадии раскладываются по их свободной памяти.\n",
        ROLE_SHARD);
}

static int cmd_info(dc_config *cfg, llm_model *m, char *err, size_t errlen)
{
    llm_node nodes[DC_MAX_EPS];
    int      owner[512];
    int      nn = collect_nodes(cfg, nodes, DC_MAX_EPS, err, errlen);
    if (nn < 0) return 3;

    printf("Модель: %s, слоёв %d, embd %d, голов %d/%d, ff %d, словарь %d, %zu байт\n",
           m->arch, m->n_layer, m->n_embd, m->n_head, m->n_head_kv, m->n_ff, m->n_vocab,
           m->size);
    for (int k = 0; k < nn; k++)
        printf("узел %-24s свободно %ld байт под модель, примет до %d байт\n",
               nodes[k].ep->url, nodes[k].cap, nodes[k].ep->cap.max_body);

    int nstages = 0;
    if (split_stages(m, nodes, nn, owner, &nstages, err, errlen) != 0) {
        printf("\nраскладка невозможна: %s\n", err);
        return 4;
    }
    print_layout(m, nodes, nn, owner, nstages);
    return 0;
}

static int cmd_deploy(dc_config *cfg, llm_model *m, char *err, size_t errlen)
{
    llm_node nodes[DC_MAX_EPS];
    int      owner[512];
    int      nn = collect_nodes(cfg, nodes, DC_MAX_EPS, err, errlen);
    if (nn < 0) return 3;
    int nstages = 0;
    if (split_stages(m, nodes, nn, owner, &nstages, err, errlen) != 0) {
        printf("\nраздать нечего: %s\n", err);
        return 4;
    }
    print_layout(m, nodes, nn, owner, nstages);

    int sock = udp_socket_client(err, errlen);
    if (sock < 0) return 3;

    pipeline P = { .sock = sock, .m = m, .nodes = nodes, .owner = owner,
                   .nstages = nstages, .n_embd = m->n_embd };
    for (int s = 0; s < nstages; s++) {
        printf("стадия %d -> %s\n", s, nodes[owner[s]].ep->url);
        if (pipe_deploy_stage(&P, s) != 0) {
            close(sock);
            return 1;
        }
    }
    close(sock);
    printf("\nВеса разложены. Дальше: dcctl llm prompt %s \"текст\"\n", "МОДЕЛЬ");
    return 0;
}

static int cmd_prompt(dc_config *cfg, llm_model *m, int argc, char **argv, char *err,
                      size_t errlen)
{
    const char *prompt = argc > 0 ? argv[0] : "distcomp";
    int   ngen = argc > 1 ? atoi(argv[1]) : 8;
    if (ngen < 1) ngen = 1;
    if (ngen > 256) ngen = 256;

    /* промпт -> токены */
    int ids[4096];
    int nid = llm_tok_encode(&m->tok, prompt, ids, (int)(sizeof ids / sizeof ids[0]), err, errlen);
    if (nid < 0) return 2;

    char text[4096];
    llm_tok_decode(&m->tok, ids, nid, text, sizeof text);
    printf("Промпт: «%s»\n", text);
    printf("Токенов в промпте: %d\n", nid);

    llm_node nodes[DC_MAX_EPS];
    int      owner[512];
    int      nn = collect_nodes(cfg, nodes, DC_MAX_EPS, err, errlen);
    if (nn < 0) return 3;
    int nstages = 0;
    if (split_stages(m, nodes, nn, owner, &nstages, err, errlen) != 0) {
        printf("\nвыполнить нечем: %s\n", err);
        return 4;
    }
    print_layout(m, nodes, nn, owner, nstages);

    int sock = udp_socket_client(err, errlen);
    if (sock < 0) return 3;

    pipeline P = { .sock = sock, .m = m, .nodes = nodes, .owner = owner,
                   .nstages = nstages, .n_embd = m->n_embd,
                   .state = xcalloc((size_t)m->n_embd + 8, sizeof(float)) };
    for (int s = 0; s < nstages; s++)
        if (pipe_deploy_stage(&P, s) != 0) {
            free(P.state);
            close(sock);
            return 1;
        }

    /* локальный эталон */
    local_run L;
    if (local_open(&L, m, err, errlen) != 0) {
        free(P.state);
        close(sock);
        return 1;
    }

    printf("\nГенерация (жадная):\n\n");
    int bad = 0;
    int tok = 0;
    int last = 0;

    /* промпт: и эталон, и конвейер прогоняют его целиком и копят контекст.
     * Выход последнего шага промпта — это первый сгенерированный токен. */
    for (int i = 0; i < nid; i++) {
        tok = ids[i];
        if (local_forward(&L, tok) < 0) {
            fprintf(stderr, "эталон не смог обработать токен %d\n", tok);
            local_free(&L); free(P.state); close(sock);
            return 1;
        }
        int got = pipe_step(&P, tok, err, errlen);
        if (got < 0) {
            fprintf(stderr, "конвейер не смог обработать токен %d: %s\n", tok, err);
            local_free(&L); free(P.state); close(sock);
            return 1;
        }
        printf("%d ", tok);
        fflush(stdout);
        last = got;
    }
    tok = nid > 0 ? last : 0;

    /* генерация: печатаем токен, затем сверяем шаг обоих прогонов на одном входе */
    for (int i = 0; i < ngen; i++) {
        printf("%d ", tok);
        fflush(stdout);
        int ref = local_forward(&L, tok);
        int got = pipe_step(&P, tok, err, errlen);
        if (got < 0) {
            fprintf(stderr, "\nконвейер не смог обработать токен %d: %s\n", tok, err);
            local_free(&L); free(P.state); close(sock);
            return 1;
        }
        if (got != ref) bad++;
        tok = got;
    }
    printf("\n");

    local_free(&L);
    free(P.state);
    close(sock);

    if (bad) {
        printf("\nСверка: расхождение в %d токенах из %d — конвейер считает\n"
               "не так, же как целиком на планировщике.\n", bad, ngen);
        return 1;
    }
    printf("\nСверка: все %d сгенерированных токенов совпали с локальным прогоном.\n", ngen);
    return 0;
}

int shard_cmd(dc_config *cfg, int argc, char **argv, char *err, size_t errlen)
{
    if (argc < 1) {
        usage();
        return 2;
    }
    const char *sub = argv[0];
    if (argc < 2) {
        usage();
        return 2;
    }
    const char *path = argv[1];

    llm_model m;
    /* ограничения узлов (max_body и free_mem) приходят из PONG: без опроса
     * раскладка была бы вслепую */
    if (sched_learn_caps(cfg) == 0) {
        snprintf(err, errlen,
                 "не удалось опросить узлы роли %s: они не отвечают или не сообщили "
                 "свободной памяти (free_mem)",
                 ROLE_SHARD);
        return 3;
    }
    if (llm_open_file(&m, path, err, errlen) != 0) return 3;

    int rc;
    if (strcmp(sub, "info") == 0) {
        rc = cmd_info(cfg, &m, err, errlen);
    } else if (strcmp(sub, "deploy") == 0) {
        rc = cmd_deploy(cfg, &m, err, errlen);
    } else if (strcmp(sub, "prompt") == 0) {
        rc = cmd_prompt(cfg, &m, argc - 2, argv + 2, err, errlen);
    } else {
        snprintf(err, errlen, "неизвестная подкоманда llm: %s", sub);
        rc = 2;
    }
    llm_free(&m);
    return rc;
}