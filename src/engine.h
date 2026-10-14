#ifndef DC_ENGINE_H
#define DC_ENGINE_H

#include "gguf.h"

#include <stddef.h>

/*
 * Движок llama-архитектуры (LLaMA, Mistral, Qwen2 и родственники):
 * RMSNorm, RoPE, grouped-query attention, SwiGLU, KV-кэш.
 *
 * Модель хранится ровно так, как лежит в файле (F32 или F16). Ни одна стадия
 * не преобразует веса: одна и та же модель целиком и та же модель, разложенная
 * по устройствам, дают одинаковые числа — иначе сверка была бы сравнением
 * двух разных моделей.
 *
 * Узел держит либо вход (эмбеддинг), либо диапазон слоёв, либо выход (норма и
 * голова). Полный набор таких узлов — это LLM, работающая в памяти парка
 * устройств.
 */

/* Что именно лежит на этом узле */
enum {
    LLM_STAGE_EMBED = 0,   /* token_embd — вход */
    LLM_STAGE_LAYERS,      /* blk.first..blk.last */
    LLM_STAGE_HEAD,        /* output_norm + output — выход, здесь же выбор токена */
};

typedef struct {
    const gguf_tensor *t;
} llm_w;

typedef struct {
    llm_w attn_norm, attn_q, attn_k, attn_v, attn_out;
    llm_w ffn_norm, ffn_gate, ffn_up, ffn_down;
} llm_layer;

typedef struct {
    /* геометрия модели */
    int    n_layer, n_embd, n_head, n_head_kv, n_ff, head_dim, n_vocab, n_rep;
    float  rms_eps, rope_base;
    int    n_ctx;

    /* что обслуживает этот узел */
    int    stage;
    int    first, last;

    llm_w  w_emb, w_norm, w_out;
    llm_layer *layers;

    /* рабочее место: всё это — куски арены, malloc внутри нет */
    float *xb, *xb2, *xb3;        /* [n_embd], [max(n_embd,n_ff)], [n_ff] */
    float *wn, *xs;               /* [n_embd] веса норм, [max(n_ff,n_embd)] черновик */
    float *q, *att;               /* [n_head*head_dim] ×2 */
    float *k, *v;                 /* [n_kv*head_dim] ×2 */
    float *qh;                     /* [head_dim] — одна голова после RoPE */
    float *scores;                 /* [n_ctx] — очки внимания до softmax */
    float *logits;                 /* [n_vocab], только для HEAD */
    float *kc, *vc;                /* KV-кэш: [слоёв][n_ctx][n_kv*head_dim] */
    int    pos;                    /* сколько токенов уже в кэше */
} llm_ctx;

/* Сколько float нужно под эту часть модели при контексте n_ctx
 * (n_ctx = 0 — взять из самой модели). */
size_t llm_ctx_floats(const gguf_file *g, int stage, int first, int last, int n_ctx);

/* Собирает контекст: n_ctx = 0 — длина контекста из модели, иначе своя
 * (на устройстве разумно просить меньше, чем просит модель: KV-кэш памятен).
 * Буферы выделяются из arena, туда же — KV-кэш. */
int  llm_ctx_open(llm_ctx *c, const gguf_file *g, int stage, int first, int last,
                  int n_ctx, llm_layer *layer_store, float *arena, size_t arena_floats,
                  char *err, size_t errlen);

/* Один шаг генерации: token -> состояние.
 *  - EMBED:  out[0..n_embd)        — эмбеддинг токена
 *  - LAYERS: out на входе и выходе — состояние после своих слоёв
 *  - HEAD:   out[0..n_embd) — состояние, out[n_embd] — выбранный токен
 * Возвращает 0 в порядке. */
int  llm_forward(llm_ctx *c, int token, float *out);

void llm_reset(llm_ctx *c);

/* Прогоняет ровно один слой layer_idx (0..last-first) на текущей позиции,
 * не сдвигая её. Нужен для пошаговой сверки и для режима «слой за слоем». */
int  llm_layer_once(llm_ctx *c, int layer_idx, float *out);

/* Прогоняет первые n слоёв этой части модели (n = 0 — все).
 * Позиция в кэше сдвигается на один токен за вызов. */
int  llm_step_layers(llm_ctx *c, int n, float *out);

/* Читает логиты, посчитанные последним вызовом HEAD. */
int  llm_pick(const llm_ctx *c);

#endif /* DC_ENGINE_H */