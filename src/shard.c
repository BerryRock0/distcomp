#include "shard.h"
#include "engine.h"
#include "gguf.h"
#include "json.h"
#include "sched.h"
#include "tok.h"
#include "udp.h"
#include "util.h"
#include "xport.h"

#include <errno.h>
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

/* Сколько токенов контекста готовим по умолчанию и сколько в принципе можно.
 * KV-кэш растёт линейно по контексту, поэтому это в первую очередь вопрос
 * памяти устройств, а не «хочется ли нам длинного ответа». */
#define DC_CTX_DEFAULT 512
#define DC_CTX_MAX     4096

/* Сколько токенов промпта мы в состоянии закодировать (потолок для буфера). */
#define DC_PROMPT_TOKENS 8192

/* Сколько байт модели читать ради словаря. Заголовок со словарём в сотни
 * тысяч слов и списком слияний занимает десятки мегабайт; веса после него
 * не нужны, поэтому читаем начало файла, а не весь. */
#define DC_TOKENIZER_READ_MAX (128 * 1024 * 1024)

/* Таймаут одного обмена с узлом конвейера. Кусок тензора не больше max_body,
 * поэтому этого хватает и на медленной плате. */
#define DC_PIPE_TIMEOUT_MS 20000

/* Сколько раз повторять запрос, прежде чем признать узел недоступным.
 * Раздача модели — это тысячи запросов подряд, и UDP их часть теряет;
 * три попытки подряд на одной потерянной дейтаграмме — обычное дело. */
#define DC_PIPE_ATTEMPTS 4

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

static int llm_open_file(llm_model *m, const char *path, int n_ctx_want,
                         char *err, size_t errlen)
{
    memset(m, 0, sizeof *m);
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errlen, "не удалось открыть %s: %s", path, strerror(errno)); return -1; }
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

    /* Квантованные веса движок теперь считает: gguf_dequant разворачивает
     * блоки Q2_K..Q8_K и старые кванты в float перед умножением, поэтому
     * отказывать модели тут больше не из-за чего. */
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

    /* Сколько токенов контекста мы собираемся прогнать. Это не берётся из
     * модели: у Qwen2.5 в метаданных 32768, а плата столько не удержит —
     * контекст это KV-кэш, и он памятен. Поэтому задаётся явно, а сверху
     * ограничен DC_CTX_MAX, иначе одна опечатка в конфигурации уронит парк
     * устройств попыткой выделить кубом гигабайт. */
    if (n_ctx_want > DC_CTX_MAX) n_ctx_want = DC_CTX_MAX;
    if (n_ctx_want > 0) m->n_ctx = n_ctx_want;
    if (m->n_ctx <= 0 || m->n_ctx > DC_CTX_MAX) m->n_ctx = DC_CTX_DEFAULT;

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
                        stage_tensor *out, int max, size_t *total_bytes,
                         char *eout, size_t eoutlen)
{
    *total_bytes = 0;
    int n = 0;
    const char *name;

#define ADD(nn)                                                            \
    do {                                                                   \
        const gguf_tensor *t = gguf_tensor_find(&m->g, nn);                \
        if (!t) {                                                          \
            snprintf(err_tmp, sizeof err_tmp, "в модели нет тензор %s", nn); \
            if (eout) snprintf(eout, eoutlen, "%s", err_tmp);                \
            return -1;                                                     \
        }                                                                  \
        /* out == NULL — считаем только вес стадии. Имена не нужны, и тогда \
         * количество тензоров не ограничено буфером: у 28 слоёв Qwen2 их \
         * 28x9 = 252, и раньше стадия слоёв молча возвращала -1. */       \
        if (out) {                                                         \
            if (n >= max) {                                                 \
                if (eout) snprintf(eout, eoutlen,                        \
                                   "стадия не влезает в список: нужно больше %d тензоров", \
                                   max);                                   \
                return -1;                                                 \
            }                                                               \
            snprintf(out[n].name, sizeof out[n].name, "%s", nn);           \
            out[n].nbytes = t->nbytes;                                     \
            out[n].t = t;                                                  \
        }                                                                  \
        *total_bytes += t->nbytes;                                         \
        n++;                                                               \
    } while (0)

/* Как ADD, но отсутствие тензора не ошибка: так добавляются смещения,
 * которых нет у моделей без них. */
#define ADD_OPT(nn)                                                           \
    do {                                                                      \
        const gguf_tensor *t = gguf_tensor_find(&m->g, nn);                   \
        if (!t) break;                                                        \
        if (out) {                                                            \
            if (n >= max) {                                                    \
                if (eout) snprintf(eout, eoutlen,                              \
                                   "стадия не влезает в список: нужно больше %d тензоров", \
                                   max);                                       \
                return -1;                                                     \
            }                                                                  \
            snprintf(out[n].name, sizeof out[n].name, "%s", nn);              \
            out[n].nbytes = t->nbytes;                                         \
            out[n].t = t;                                                      \
        }                                                                      \
        *total_bytes += t->nbytes;                                             \
        n++;                                                                   \
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
                              "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                              /* смещения есть не у всех моделей: у Qwen2 у Q, K и V
                               * они есть, и без них слой считается неверно */
                              "attn_q.bias", "attn_k.bias", "attn_v.bias" };
        /* веса обязательны, смещения — нет: ADD_OPT их пропускает */
        for (int i = first; i <= last; i++) {
            for (size_t k = 0; k < 9; k++) {
                snprintf(nm, sizeof nm, "blk.%d.%s", i, sfx[k]);
                ADD(nm);
            }
            for (size_t k = 9; k < sizeof sfx / sizeof sfx[0]; k++) {
                snprintf(nm, sizeof nm, "blk.%d.%s", i, sfx[k]);
                ADD_OPT(nm);
            }
        }
    }
    return n;
#undef ADD
#undef ADD_OPT
}

/* Ограничение на кусок данных: base64 распухает в 4/3, плюс заголовок
 * команды и запас на ответ.
 *
 * Запас в 1/32 оставлен не просто так: дейтаграмма впритык к
 * DC_UDP_MAX_DGRAM (заголовок UDP + JSON-обвязка) терялась по дороге, и
 * раздача вставала намертво посередине тензора. */
static long chunk_budget(const dc_endpoint *ep)
{
    long body = ep->cap.max_body > 96 ? ep->cap.max_body - 96 : 192;
    long raw = body * 3 / 4;
    raw -= raw / 32;
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
    size_t total = 0;
    /* out = NULL: нужны только байты, без списка тензоров */
    int n = stage_tensors(m, stage, first, last, NULL, 0, &total, NULL, 0);
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
    const llm_model *m;
    const llm_node  *nodes;
    const int       *owner;
    int              nstages;
    float           *state;      /* текущее состояние, n_embd */
    int              n_embd;
} pipeline;

/* Отправляет команду узлу: оборачивает её в конверт узла (role=shard,
 * payload=команда), разбирает ответ и отдаёт поле output.
 * Транспорт выбирается по схеме адреса: udp://, tcp://, dtn://. */
static int pipe_send(dc_endpoint *ep, const char *payload, char **out_body,
                     char *err, size_t errlen)
{
    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\"role\":\"");
    sb_add(&b, ROLE_SHARD);
    sb_add(&b, "\",\"title\":\"\",\"payload\":");
    sb_add_json_string(&b, payload);
    sb_add(&b, "}");

    /* При раздаче модели уходит около шестнадцати тысяч запросов подряд, и
     * UDP на загруженной машине теряет часть дейтаграмм. Без повтора одна
     * потерянная прерывала всю раздачу таймаутом в двадцать секунд. */
    char *resp = NULL;
    int   rc   = -1;
    for (int attempt = 0; attempt < DC_PIPE_ATTEMPTS; attempt++) {
        if (resp) { free(resp); resp = NULL; }
        rc = xport_request_raw(ep, b.p, strlen(b.p), DC_PIPE_TIMEOUT_MS, &resp, err, errlen);
        if (rc == 0) break;
        log_info("повтор %d: узел %s не ответил (%s)", attempt + 1, ep->url, err);
    }
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

/* Отправляет один тензор на узел, дробями по вместимости одного сообщения. */
static int pipe_send_tensor(const llm_node *nd, const stage_tensor *tp, char *err,
                            size_t errlen)
{
    long budget = chunk_budget(nd->ep);
    long off = 0;
    while (off < (long)tp->nbytes) {
        long take = (long)tp->nbytes - off;
        if (take > budget) take = budget;
        long chunk = take / 3 * 3;
        if (chunk <= 0) chunk = take;

        strbuf b;
        sb_init(&b);
        sb_printf(&b, "wt|%s|%u|", tp->name, tp->t->n_dims);
        for (unsigned d = 0; d < tp->t->n_dims; d++)
            sb_printf(&b, "%s%lu", d ? "," : "", tp->t->dims[d]);
        sb_printf(&b, "|%u|%ld|%ld|", tp->t->type, off, (long)tp->t->nbytes);
        char *b64 = xmalloc(b64_encode_size((size_t)chunk) + 1);
        b64_encode(tp->t->data + off, (size_t)chunk, b64, b64_encode_size((size_t)chunk) + 1);
        sb_add(&b, b64);
        free(b64);

        char *out = NULL;
        int   rc  = pipe_send(nd->ep, b.p, &out, err, errlen);
        free(out);
        sb_free(&b);
        if (rc != 0) {
            /* сообщение узла лежит в err — переносим его в отдельный буфер,
             * иначе snprintf писал бы в перекрывающийся сам с собой аргумент */
            char why[256] = {0};
            snprintf(why, sizeof why, "%s", err);
            snprintf(err, errlen, "тензор %s не ушёл на %s: %s", tp->name, nd->ep->url, why);
            return -1;
        }
        off += chunk;
    }
    return 0;
}

/* Заливает стадию на узел: тензорами, потом shape.
 * err берётся у вызывающего: раньше ошибка писалась в локальный буфер и
 * доходила до пользователя пустой строкой («раздать нечего: »). */
static int pipe_deploy_stage(pipeline *P, int s, char *err, size_t errlen)
{
    llm_model *m = (llm_model *)P->m;
    const llm_node *nd = &P->nodes[P->owner[s]];
    size_t total;

    /* сначала освобождаем узел: вторая модель не влезет поверх первой */
    {
        char *out = NULL;
        if (pipe_send(nd->ep, "free", &out, err, errlen) != 0) {
            fprintf(stderr, "узел %s не освободился: %s\n", nd->ep->url, err);
            return -1;
        }
        free(out);
    }

    /* Идём по слоям, а не по всей стадии сразу: у 28 слоёв Qwen2 в стадии
     * слоёв 252 тензора, и список имён в один буфер не помещался. На слой
     * приходится девять записей — это помещается всегда. */
    stage_tensor st[16];
    for (int layer = nd->first; layer <= nd->last; layer++) {
        int n = stage_tensors(m, nd->stage, layer, layer, st, 16, &total, err, errlen);
        if (n < 0) return -1;
        for (int i = 0; i < n; i++)
            if (pipe_send_tensor(nd, &st[i], err, errlen) != 0) return -1;
    }

    /* shape: геометрия и стадия */
    strbuf b;
    sb_init(&b);
    sb_printf(&b, "shape|%d|%d|%d|%d|%d|%d|%d|%d|%d|%d|%.8g|%.8g|%d",
              nd->stage, nd->first, nd->last, m->n_embd, m->n_head, m->n_head_kv, m->n_ff,
              m->head_dim, m->n_vocab, m->n_layer, (double)m->rms_eps, (double)m->rope_base,
              m->n_ctx);
    char *out = NULL;
    if (pipe_send(nd->ep, b.p, &out, err, errlen) != 0) {
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
        if (pipe_send(nd->ep, b.p, &out, err, errlen) != 0) {
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
            if (pipe_send(nd->ep, b.p, &out, err, errlen) != 0) {
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

/* ------------------------------------------------------ распределённый LLM --- */
/*
 * То, ради чего всё затевалось: модель лежит не здесь. Планировщик держит
 * только текст промпта и одно состояние размера n_embd, а веса — на узлах.
 * Поэтому планировать цели можно с машины, на которой для модели памяти нет
 * вовсе: узлов столько, сколько нужно, и каждый получает ровно свою часть.
 */

struct dc_infer {
    llm_model m;
    char      path[512];

    llm_node nodes[DC_MAX_EPS];
    int      owner[DC_MAX_EPS];
    int      nn;          /* узлов роли shard */
    int      nstages;     /* стадий в конвейере */

    float *state;         /* активация между стадиями, n_embd */
    int    used;          /* токенов контекста в последней генерации */
    int    verbose;
};

static void infer_release(dc_infer *in)
{
    if (!in) return;
    llm_free(&in->m);
    free(in->state);
    free(in);
}

void dc_infer_free(dc_infer *in)
{
    infer_release(in);
}

/* Раскладывает модель по узлам, но ничего не заливает: нужно для проверки
 * «поместится ли вообще», когда веса трогать не хочется. */
static int infer_plan(dc_config *cfg, const char *model_path, const dc_infer_opts *o,
                      dc_infer **out, char *err, size_t errlen)
{
    *out = NULL;
    int n_ctx   = o ? o->n_ctx : 0;
    int verbose = o ? o->verbose : 0;

    /* раскладка вслепую бессмысленна: сколько памяти отдаёт узел и какой у него
     * лимит тела, известно только из ответа на PING */
    if (sched_learn_caps(cfg) == 0) {
        snprintf(err, errlen,
                 "не удалось опросить узлы роли %s: они не отвечают или не сообщили "
                 "свободную память (free_mem)",
                 ROLE_SHARD);
        return -1;
    }

    dc_infer *in = xcalloc(1, sizeof *in);
    if (llm_open_file(&in->m, model_path, n_ctx, err, errlen) != 0) {
        free(in);
        return -1;
    }
    snprintf(in->path, sizeof in->path, "%s", model_path);
    in->verbose = verbose;

    in->nn = collect_nodes(cfg, in->nodes, DC_MAX_EPS, err, errlen);
    if (in->nn < 0) {
        infer_release(in);
        return -1;
    }
    if (split_stages(&in->m, in->nodes, in->nn, in->owner, &in->nstages, err, errlen) != 0) {
        infer_release(in);
        return -1;
    }
    in->state = xcalloc((size_t)in->m.n_embd + 8, sizeof(float));
    if (verbose) print_layout(&in->m, in->nodes, in->nn, in->owner, in->nstages);
    *out = in;
    return 0;
}

/* Заливает веса во все стадии. Собственно подготовка к генерации. */
int dc_infer_open(dc_config *cfg, const char *model_path, const dc_infer_opts *o,
                  dc_infer **out, char *err, size_t errlen)
{
    dc_infer *in = NULL;
    if (infer_plan(cfg, model_path, o, &in, err, errlen) != 0) return -1;

    pipeline P = { .m = &in->m, .nodes = in->nodes, .owner = in->owner,
                   .nstages = in->nstages, .n_embd = in->m.n_embd, .state = in->state };
    for (int s = 0; s < in->nstages; s++) {
        if (pipe_deploy_stage(&P, s, err, errlen) != 0) {
            infer_release(in);
            return -1;
        }
    }
    *out = in;
    return 0;
}

/* Раскладка без заливки весов — для панели и для `llm info`. */
int dc_infer_layout(dc_config *cfg, const char *model_path, const dc_infer_opts *o,
                    dc_infer **out, char *err, size_t errlen)
{
    return infer_plan(cfg, model_path, o, out, err, errlen);
}

/* Сбрасывает KV-кэш на всех стадиях: узлы помнят контекст между вызовами,
 * а каждый новый промпт начинается с чистого листа. */
static int infer_reset(dc_infer *in, char *err, size_t errlen)
{
    for (int s = 0; s < in->nstages; s++) {
        char *out = NULL;
        if (pipe_send(in->nodes[in->owner[s]].ep, "reset", &out, err, errlen) != 0) return -1;
        free(out);
    }
    return 0;
}

char *dc_infer_generate(dc_infer *in, const char *prompt, int max_new,
                        char *err, size_t errlen)
{
    return dc_infer_generate_cb(in, prompt, max_new, NULL, NULL, err, errlen);
}

/* Граница UTF-8 в накопленном тексте: у BPE один токен — это байт или часть
 * многобайтного знака, поэтому последний токен иногда неполон, и отдавать его
 * в браузер рано. Возвращает длину безопасного префикса. */
static size_t utf8_safe(const char *s, size_t n)
{
    if (!n) return 0;
    /* идём от конца к началу последнего знака */
    size_t i = n;
    while (i > 0 && ((unsigned char)s[i - 1] & 0xC0) == 0x80) i--;
    if (i == 0) return 0;                       /* хвост без начала — ждём */
    unsigned char lead = (unsigned char)s[i - 1];
    size_t need = lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
    return (n - i + 1) >= need ? n : i - 1;
}

/* Генерация с потоком: на каждый новый токен вызывается cb с его текстом.
 * cb возвращает 0, чтобы остановиться; если cb нет, токены просто копятся.
 * Весь ответ собирается и возвращается в любом случае — планировщику он и
 * нужен целиком. */
char *dc_infer_generate_cb(dc_infer *in, const char *prompt, int max_new,
                           dc_tok_cb cb, void *ud, char *err, size_t errlen)
{
    if (!in || !prompt) {
        snprintf(err, errlen, "конвейер не открыт");
        return NULL;
    }

    int *ids = xmalloc(sizeof(int) * DC_PROMPT_TOKENS);
    int  nid = llm_tok_encode(&in->m.tok, prompt, ids, DC_PROMPT_TOKENS, err, errlen);
    if (nid < 0) {
        free(ids);
        return NULL;
    }

    /* Позиция в кэше нумеруется с нуля, поэтому для промпта и ответа вместе
     * нужно на один токен меньше, чем объявленный контекст. */
    int budget = in->m.n_ctx - 1;
    if (nid > budget) {
        snprintf(err, errlen,
                 "промпт не помещается в контекст: %d токенов нужно, узлы держат %d. "
                 "Поднимите n_ctx в секции [llm] — но тогда каждому узлу нужно "
                 "больше памяти под KV-кэш — или сократите цель.",
                 nid, in->m.n_ctx);
        free(ids);
        return NULL;
    }
    if (max_new <= 0) max_new = 256;
    if (max_new > budget - nid) max_new = budget - nid;
    if (max_new <= 0) {
        snprintf(err, errlen, "контекста не хватает даже на один токен ответа (%d)", budget);
        free(ids);
        return NULL;
    }

    if (infer_reset(in, err, errlen) != 0) {
        free(ids);
        return NULL;
    }

    pipeline P = { .m = &in->m, .nodes = in->nodes, .owner = in->owner,
                   .nstages = in->nstages, .n_embd = in->m.n_embd, .state = in->state };

    /* Промпт прогоняется целиком: он не генерируется, а читается, и его выход
     * — это вход состояния. Последний шаг промпта даёт первый новый токен. */
    int next = 0;
    for (int i = 0; i < nid; i++) {
        next = pipe_step(&P, ids[i], err, errlen);
        if (next < 0) {
            free(ids);
            return NULL;
        }
    }
    free(ids);

    int  eos   = in->m.tok.eos_id;
    int *gen   = xcalloc((size_t)max_new + 1, sizeof(int));
    int  ngen  = 0;
    int  token = nid > 0 ? next : 0;

    /* уже отданный в поток текст: с каждым токеном отдаём только продолжение */
    size_t sent = 0;
    char  *acc  = xmalloc(1);
    acc[0] = '\0';

    for (int i = 0; i < max_new; i++) {
        if (token == eos) break;
        gen[ngen++] = token;
        if (cb) {
            /* пересобираем весь ответ: разбор BPE может «дописать» знак,
             * поэтому кусок считаем от конца, а не от текущего токена */
            free(acc);
            size_t cap = (size_t)ngen * 8 + 64;
            acc = xmalloc(cap);
            llm_tok_decode(&in->m.tok, gen, ngen, acc, cap);
            size_t have = strlen(acc);
            size_t upto = utf8_safe(acc, have);
            if (upto > sent) {
                int cont = cb(token, acc + sent, ud);
                sent = upto;
                if (!cont) break;
            }
        }
        next = pipe_step(&P, token, err, errlen);
        if (next < 0) {
            free(acc);
            free(gen);
            return NULL;
        }
        token = next;
    }
    in->used = nid + ngen;

    /* Токен на текст даёт примерно 4 байта для кириллицы и 1 для ASCII;
     * берём с запасом и всё равно проверяем, что влезло. */
    size_t cap = (size_t)ngen * 8 + 64;
    char  *text = xmalloc(cap);
    llm_tok_decode(&in->m.tok, gen, ngen, text, cap);
    free(acc);
    free(gen);
    return text;
}

int dc_infer_used(const dc_infer *in)
{
    return in ? in->used : 0;
}

const char *dc_infer_model_path(const dc_infer *in) { return in ? in->path : ""; }
const char *dc_infer_arch(const dc_infer *in)       { return in && in->m.arch ? in->m.arch : ""; }
int  dc_infer_n_layer(const dc_infer *in)           { return in ? in->m.n_layer : 0; }
int  dc_infer_n_embd(const dc_infer *in)            { return in ? in->m.n_embd : 0; }
int  dc_infer_n_vocab(const dc_infer *in)           { return in ? in->m.n_vocab : 0; }
int  dc_infer_n_ctx(const dc_infer *in)             { return in ? in->m.n_ctx : 0; }
int  dc_infer_n_nodes(const dc_infer *in)           { return in ? in->nn : 0; }
int  dc_infer_n_stages(const dc_infer *in)         { return in ? in->nstages : 0; }
size_t dc_infer_model_bytes(const dc_infer *in)     { return in ? in->m.size : 0; }

/* Размер словаря в байтах — насколько «жирный» файл модели. */
int dc_infer_vocab_open(const dc_infer *in)
{
    return in && in->m.n_vocab > 0 ? 1 : 0;
}

void dc_infer_print_layout(const dc_infer *in)
{
    if (!in) return;
    print_layout(&in->m, in->nodes, in->nn, in->owner, in->nstages);
}

/* Сколько токенов в тексте — см. объявление в shard.h. */
int llm_model_tokens(const char *model_path, const char *text, int *out,
                     char *err, size_t errlen)
{
    if (!out) return -1;
    *out = 0;
    if (!model_path || !text) return -1;

    /* Заголовок со словарём весит единицы мегабайт даже у модели на 150 тысяч
     * слов. Веса после него планировщику не нужны, поэтому читаем начало
     * файла, а не весь: на трёхгигабайтной модели это разница между
     * «секунда» и «вечность». */
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        snprintf(err, errlen, "не удалось открыть %s: %s", model_path, strerror(errno));
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize <= 0) {
        snprintf(err, errlen, "файл %s пуст", model_path);
        fclose(f);
        return -1;
    }

    long want = fsize;
    if (want > DC_TOKENIZER_READ_MAX) want = DC_TOKENIZER_READ_MAX;
    unsigned char *head = xmalloc((size_t)want);
    size_t         got  = fread(head, 1, (size_t)want, f);
    fclose(f);
    if (got == 0) {
        free(head);
        snprintf(err, errlen, "%s не читается", model_path);
        return -1;
    }

    gguf_file g;
    char      perr[256] = {0};
    /* Только заголовок: веса за его пределами в этот кусок не попали и не
     * нужны. Обычный gguf_open на таком префиксе отверг бы файл целиком —
     * он ждёт все веса, а мы их не читали. */
    if (gguf_open_head(&g, head, got, perr, sizeof perr) != 0) {
        free(head);
        snprintf(err, errlen,
                 "не удалось прочитать словарь модели: %s "
                 "(заголовок со словарём не поместился в %ld байт из %ld?)",
                 perr, got, fsize);
        return -1;
    }

    llm_tok tok;
    int     rc = llm_tok_open(&tok, &g, perr, sizeof perr);
    if (rc == 0) {
        static int ids[DC_PROMPT_TOKENS];
        *out = llm_tok_encode(&tok, text, ids, DC_PROMPT_TOKENS, perr, sizeof perr);
        llm_tok_close(&tok);
    }
    gguf_close(&g);
    free(head);

    if (rc != 0) {
        snprintf(err, errlen, "токенизатор: %s", perr);
        return -1;
    }
    if (*out < 0) {
        snprintf(err, errlen, "текст не помещается в %d токенов", DC_PROMPT_TOKENS);
        *out = 0;
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------- команды --- */

static void usage(void)
{
    printf(
        "dcctl llm — LLM, разложенная по устройствам\n"
        "\n"
        "  llm info [ФАЙЛ]                          раскладка по узлам\n"
        "  llm deploy [ФАЙЛ]                        раздать веса по узлам\n"
        "  llm prompt [ФАЙЛ] \"текст\" [токенов]      генерация с проверкой\n"
        "  llm check [ФАЙЛ]                         готов ли планировщик\n"
        "\n"
        "ФАЙЛ — модель GGUF (F32/F16, архитектура llama). Если не указан, берётся\n"
        "model_file из секции [llm]. Узлы — из [role %s]: стадии раскладываются по\n"
        "их свободной памяти, по сети идёт активация, а не модель.\n",
        ROLE_SHARD);
}

/* Модель из аргумента, иначе из конфигурации. */
static const char *model_arg(const dc_config *cfg, int argc, char **argv, char *err,
                             size_t errlen)
{
    if (argc > 0 && argv[0] && argv[0][0] && argv[0][0] != '-') return argv[0];
    if (cfg->model_file && *cfg->model_file) return cfg->model_file;
    snprintf(err, errlen,
             "не задана модель: укажите файл аргументом или model_file в секции [llm]");
    return NULL;
}

static dc_infer_opts opts_from(const dc_config *cfg, int verbose)
{
    dc_infer_opts o = { .n_ctx = cfg->n_ctx, .verbose = verbose };
    return o;
}

static int cmd_info(dc_config *cfg, int argc, char **argv, char *err, size_t errlen)
{
    const char *path = model_arg(cfg, argc, argv, err, errlen);
    if (!path) {
        printf("%s\n", err);
        return 2;
    }

    /* info ничего не заливает: спрашиваются только PING */
    dc_infer      *in = NULL;
    dc_infer_opts  o  = opts_from(cfg, 0);
    if (dc_infer_layout(cfg, path, &o, &in, err, errlen) != 0) {
        printf("\nраскладка невозможна: %s\n", err);
        return 4;
    }

    printf("Модель: %s, слоёв %d, embd %d, слов %d, контекст %d токенов, %zu байт\n",
           dc_infer_arch(in), dc_infer_n_layer(in), dc_infer_n_embd(in),
           dc_infer_n_vocab(in), dc_infer_n_ctx(in), dc_infer_model_bytes(in));
    printf("узлов роли %s: %d\n", ROLE_SHARD, dc_infer_n_nodes(in));
    dc_infer_print_layout(in);
    dc_infer_free(in);
    return 0;
}

static int cmd_deploy(dc_config *cfg, int argc, char **argv, char *err, size_t errlen)
{
    const char *path = model_arg(cfg, argc, argv, err, errlen);
    if (!path) {
        printf("%s\n", err);
        return 2;
    }

    dc_infer      *in = NULL;
    dc_infer_opts  o  = opts_from(cfg, 1);
    if (dc_infer_open(cfg, path, &o, &in, err, errlen) != 0) {
        printf("\nраздать нечего: %s\n", err);
        return 4;
    }
    printf("\nВеса разложены по %d узлам. Дальше: dcctl llm prompt \"%s\" \"текст\"\n",
           dc_infer_n_nodes(in), path);
    dc_infer_free(in);
    return 0;
}

/* Генерация с обязательной сверкой: та же модель считается целиком на
 * планировщике, и каждый токен сравнивается с тем, что вернул конвейер.
 * Расхождение означает, что где-то потерялись веса или активация. */
static int cmd_prompt(dc_config *cfg, int argc, char **argv, char *err, size_t errlen)
{
    const char *path   = model_arg(cfg, argc, argv, err, errlen);
    const char *prompt = (argc > 1 && argv[1]) ? argv[1] : "distcomp";
    int         ngen   = argc > 2 ? atoi(argv[2]) : 8;
    if (!path) {
        printf("%s\n", err);
        return 2;
    }
    if (ngen < 1) ngen = 1;
    if (ngen > 256) ngen = 256;

    dc_infer      *in = NULL;
    dc_infer_opts  o  = opts_from(cfg, 1);
    if (dc_infer_open(cfg, path, &o, &in, err, errlen) != 0) {
        printf("\nвыполнить нечем: %s\n", err);
        return 4;
    }

    /* промпт -> токены */
    int ids[DC_PROMPT_TOKENS];
    int nid = llm_tok_encode(&in->m.tok, prompt, ids, DC_PROMPT_TOKENS, err, errlen);
    if (nid < 0) {
        dc_infer_free(in);
        return 2;
    }
    if (nid > in->m.n_ctx - 1) {
        fprintf(stderr,
                "промпт не помещается в контекст: %d токенов нужно, узлы держат %d\n", nid,
                in->m.n_ctx);
        dc_infer_free(in);
        return 2;
    }

    char text[4096];
    llm_tok_decode(&in->m.tok, ids, nid, text, sizeof text);
    printf("Промпт: «%s»\n", text);
    printf("Токенов в промпте: %d\n", nid);

    /* эталон: та же модель целиком, на планировщике */
    local_run L;
    if (local_open(&L, &in->m, err, errlen) != 0) {
        fprintf(stderr, "эталон не собрался: %s\n", err);
        dc_infer_free(in);
        return 1;
    }

    pipeline P = { .m = &in->m, .nodes = in->nodes, .owner = in->owner,
                   .nstages = in->nstages, .n_embd = in->m.n_embd, .state = in->state };

    printf("\nГенерация (жадная):\n\n");
    int bad  = 0;
    int last = 0;

    /* Промпт прогоняют оба: он копит контекст, и выход его последнего шага —
     * первый сгенерированный токен. */
    for (int i = 0; i < nid; i++) {
        if (local_forward(&L, ids[i]) < 0) {
            fprintf(stderr, "эталон не смог обработать токен %d\n", ids[i]);
            local_free(&L);
            dc_infer_free(in);
            return 1;
        }
        int got = pipe_step(&P, ids[i], err, errlen);
        if (got < 0) {
            fprintf(stderr, "конвейер не смог обработать токен %d: %s\n", ids[i], err);
            local_free(&L);
            dc_infer_free(in);
            return 1;
        }
        printf("%d ", ids[i]);
        fflush(stdout);
        last = got;
    }
    int tok = nid > 0 ? last : 0;

    for (int i = 0; i < ngen; i++) {
        printf("%d ", tok);
        fflush(stdout);
        int ref = local_forward(&L, tok);
        int got = pipe_step(&P, tok, err, errlen);
        if (got < 0) {
            fprintf(stderr, "\nконвейер не смог обработать токен %d: %s\n", tok, err);
            local_free(&L);
            dc_infer_free(in);
            return 1;
        }
        if (got != ref) bad++;
        tok = got;
    }
    printf("\n");

    local_free(&L);
    dc_infer_free(in);

    if (bad) {
        printf("\nСверка: расхождение в %d токенах из %d — конвейер считает\n"
               "не так, же как целиком на планировщике.\n", bad, ngen);
        return 1;
    }
    printf("\nСверка: все %d сгенерированных токенов совпали с локальным прогоном.\n", ngen);
    return 0;
}

/* check: то же, о чём думает планировщик, когда строит план. */
static int cmd_check(dc_config *cfg, int argc, char **argv, char *err, size_t errlen)
{
    const char *path = model_arg(cfg, argc, argv, err, errlen);
    if (!path) {
        printf("%s\n", err);
        return 3;
    }

    printf("модель: %s\n", path);
    printf("узлы:   %s\n", ROLE_SHARD);

    dc_infer      *in = NULL;
    dc_infer_opts  o  = opts_from(cfg, 0);
    if (dc_infer_layout(cfg, path, &o, &in, err, errlen) != 0) {
        printf("\nготово: нет — %s\n", err);
        return 3;
    }

    printf("\nархитектура:    %s\n", dc_infer_arch(in));
    printf("слоёв:          %d\n", dc_infer_n_layer(in));
    printf("размер вектора: %d\n", dc_infer_n_embd(in));
    printf("словарь:        %d\n", dc_infer_n_vocab(in));
    printf("контекст:       %d токенов\n", dc_infer_n_ctx(in));
    printf("размер файла:   %zu байт\n", dc_infer_model_bytes(in));
    dc_infer_print_layout(in);
    printf("готово: план можно строить локально, ключи и внешний сервер не нужны.\n");
    dc_infer_free(in);
    return 0;
}

int shard_cmd(dc_config *cfg, int argc, char **argv, char *err, size_t errlen)
{
    if (argc < 1 || !argv[0]) {
        usage();
        return 2;
    }
    const char *sub = argv[0];

    if (strcmp(sub, "info") == 0)   return cmd_info(cfg, argc - 1, argv + 1, err, errlen);
    if (strcmp(sub, "deploy") == 0) return cmd_deploy(cfg, argc - 1, argv + 1, err, errlen);
    if (strcmp(sub, "prompt") == 0) return cmd_prompt(cfg, argc - 1, argv + 1, err, errlen);
    if (strcmp(sub, "check") == 0)  return cmd_check(cfg, argc - 1, argv + 1, err, errlen);

    snprintf(err, errlen, "неизвестная подкоманда llm: %s", sub);
    usage();
    return 2;
}