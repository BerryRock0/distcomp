/*
 * Проверка движка: та же модель и те же токены, что в эталоне
 * scripts/llama_ref.py. Печатает логиты и выбранные токены — их сравнивает
 * тест.
 */
#include "engine.h"
#include "gguf.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "model.gguf";
    int ngen = argc > 2 ? atoi(argv[2]) : 4;

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "нет файла %s\n", path); return 2; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "короткий файл\n"); return 2; }
    fclose(f);

    char      err[512] = {0};
    gguf_file g;
    if (gguf_open(&g, buf, (size_t)sz, err, sizeof err) != 0) {
        fprintf(stderr, "GGUF: %s\n", err);
        return 2;
    }

    const char  *arch = gguf_str(&g, "general.architecture");
    unsigned long nl = 0;
    gguf_u32(&g, &nl, "block_count", arch);
    int nlayers = (int)nl;

    /* три части модели: вход, все слои, выход — ровно как разложились бы
     * по трём устройствам */
    size_t need_emb = llm_ctx_floats(&g, LLM_STAGE_EMBED, 0, 0, 0);
    size_t need_lay = llm_ctx_floats(&g, LLM_STAGE_LAYERS, 0, nlayers - 1, 64);
    size_t need_head = llm_ctx_floats(&g, LLM_STAGE_HEAD, 0, 0, 0);

    float *a_emb = xcalloc(need_emb ? need_emb : 1, sizeof(float));
    float *a_lay = xcalloc(need_lay ? need_lay : 1, sizeof(float));
    float *a_head = xcalloc(need_head ? need_head : 1, sizeof(float));

    llm_ctx emb, lay, head;
    if (llm_ctx_open(&emb, &g, LLM_STAGE_EMBED, 0, 0, 0, NULL, a_emb, need_emb, err, sizeof err) != 0 ||
        llm_ctx_open(&lay, &g, LLM_STAGE_LAYERS, 0, nlayers - 1, 64, NULL, a_lay, need_lay, err, sizeof err) != 0 ||
        llm_ctx_open(&head, &g, LLM_STAGE_HEAD, 0, 0, 0, NULL, a_head, need_head, err, sizeof err) != 0) {
        fprintf(stderr, "сборка модели: %s\n", err);
        return 2;
    }

    float *state = xcalloc((size_t)emb.n_embd + 8, sizeof(float));

    printf("geom n_layer=%d n_embd=%d n_head=%d n_head_kv=%d n_ff=%d n_vocab=%d head_dim=%d\n",
           emb.n_layer, emb.n_embd, emb.n_head, emb.n_head_kv, emb.n_ff, emb.n_vocab, emb.head_dim);

    int prompt = 1;
    printf("prompt %d\n", prompt);
    llm_reset(&emb); llm_reset(&lay); llm_reset(&head);

    if (llm_forward(&emb, prompt, state) != 0 ||
        llm_forward(&lay, prompt, state) != 0 ||
        llm_forward(&head, prompt, state) != 0) {
        fprintf(stderr, "форвард не прошёл\n");
        return 2;
    }
    printf("logits");
    for (int i = 0; i < 8 && i < head.n_vocab; i++) printf(" %.6f", head.logits[i]);
    putchar('\n');
    printf("argmax %d\n", llm_pick(&head));

    int cur = (int)state[head.n_embd];
    printf("gen");
    for (int i = 0; i < ngen; i++) {
        printf(" %d", cur);
        if (llm_forward(&emb, cur, state) != 0) break;
        if (llm_forward(&lay, cur, state) != 0) break;
        if (llm_forward(&head, cur, state) != 0) break;
        cur = (int)state[head.n_embd];
    }
    putchar('\n');

    free(state); free(a_emb); free(a_lay); free(a_head);
    gguf_close(&g);
    free(buf);
    return 0;
}