/* Проверка токенизатора: словарь тот же, что пишет scripts/llama_ref.py,
 * поэтому эталон считается независимо на python и сверяется с C. */
#include "gguf.h"
#include "tok.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: tok_test model.gguf \"текст\"\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) return 2;
    fclose(f);

    char err[512] = {0};
    gguf_file g;
    if (gguf_open(&g, buf, (size_t)sz, err, sizeof err) != 0) { fprintf(stderr, "%s\n", err); return 2; }
    llm_tok t;
    if (llm_tok_open(&t, &g, err, sizeof err) != 0) { fprintf(stderr, "%s\n", err); return 2; }

    const char *text = argc > 2 ? argv[2] : "distcomp";
    int ids[1024];
    int n = llm_tok_encode(&t, text, ids, (int)(sizeof ids / sizeof ids[0]), err, sizeof err);
    if (n < 0) { fprintf(stderr, "encode: %s\n", err); return 2; }
    printf("токенов %d:", n);
    for (int i = 0; i < n; i++) printf(" %d", ids[i]);
    printf("\n");

    char out[4096] = {0};
    llm_tok_decode(&t, ids, n, out, sizeof out);
    printf("обратно: «%s»\n", out);
    printf("совпадает с исходным: %s\n", strcmp(out, text) == 0 ? "да" : "нет");

    /* сырые токены: первые 8 */
    printf("словарь:");
    for (int i = 0; i < 8 && (size_t)i < t.n_tokens; i++) printf(" [%s]", t.tokens[i]);
    printf("\n");
    return 0;
}