/* Печать значений по стадиям: эмбеддинг, вход каждого слоя, состояние до головы.
 * Сверяется с scripts/llama_stages.py построчно. */
#include "engine.h"
#include "gguf.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc < 3) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) return 2;
    fclose(f);

    char err[512] = {0};
    gguf_file g;
    if (gguf_open(&g, buf, (size_t)sz, err, sizeof err) != 0) { fprintf(stderr, "%s\n", err); return 2; }

    const char *arch = gguf_str(&g, "general.architecture");
    unsigned long nl = 0;
    gguf_u32(&g, &nl, "block_count", arch);
    int nlayers = (int)nl;

    size_t ne_e = llm_ctx_floats(&g, LLM_STAGE_EMBED, 0, 0, 0);
    size_t ne_l = llm_ctx_floats(&g, LLM_STAGE_LAYERS, 0, nlayers - 1, 64);
    size_t ne_h = llm_ctx_floats(&g, LLM_STAGE_HEAD, 0, 0, 0);
    llm_ctx emb, lay, head;
    if (llm_ctx_open(&emb, &g, LLM_STAGE_EMBED, 0, 0, 0, NULL, xcalloc(ne_e, sizeof(float)), ne_e, err, sizeof err) ||
        llm_ctx_open(&lay, &g, LLM_STAGE_LAYERS, 0, nlayers - 1, 64, NULL, xcalloc(ne_l, sizeof(float)), ne_l, err, sizeof err) ||
        llm_ctx_open(&head, &g, LLM_STAGE_HEAD, 0, 0, 0, NULL, xcalloc(ne_h, sizeof(float)), ne_h, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 2;
    }

    float *state = xcalloc((size_t)emb.n_embd + 8, sizeof(float));

    printf("geom n_layer=%d n_embd=%d n_head=%d n_head_kv=%d n_ff=%d n_vocab=%d head_dim=%d\n",
           emb.n_layer, emb.n_embd, emb.n_head, emb.n_head_kv, emb.n_ff, emb.n_vocab, emb.head_dim);

    for (int i = 2; i < argc; i++) {
        int tok = atoi(argv[i]);
        llm_forward(&emb, tok, state);
        printf("embed  ");
        for (int k = 0; k < 6; k++) printf("%.6f ", state[k]);
        putchar('\n');
        lay.pos = 0;
        for (int l = 0; l < nlayers; l++) {
            printf("in%-4d ", l);
            for (int k = 0; k < 6; k++) printf("%.6f ", state[k]);
            putchar('\n');
            if (llm_layer_once(&lay, l, state) != 0) return 2;
        }
        lay.pos = 1;                 /* токен целиком прошёл все слои */
        printf("final  ");
        for (int k = 0; k < 6; k++) printf("%.6f ", state[k]);
        putchar('\n');
        llm_forward(&head, tok, state);
        printf("logits ");
        for (int k = 0; k < 8; k++) printf("%.6f ", head.logits[k]);
        putchar('\n');
        printf("argmax %d\n", llm_pick(&head));
        break;                  /* достаточно первого токена: там всё видно */
    }
    return 0;
}