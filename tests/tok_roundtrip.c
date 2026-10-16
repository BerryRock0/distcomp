/* Круговая проверка токенизатора: encode -> decode возвращает исходный
 * текст, а все идентификаторы лежат в пределах словаря.
 *
 * Отличается от tok_test.c тем, что сверяет с самим собой, а не с эталоном
 * на python: удобно как быстрый тест «не развалилось ли» после правок,
 * и не требует model-файла заранее собранного скриптом.
 *
 * usage: tok_roundtrip model.gguf
 */
#include "gguf.h"
#include "tok.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(const llm_tok *t, const char *text)
{
    int ids[512];
    char err[256] = {0};
    int n = llm_tok_encode(t, text, ids, 512, err, sizeof err);
    if (n < 0) { printf("  %-40s ОШИБКА: %s\n", text, err); return 1; }

    char back[1024];
    llm_tok_decode(t, ids, n, back, sizeof back);

    int out_of_range = 0;
    for (int i = 0; i < n; i++)
        if (ids[i] < 0 || (size_t)ids[i] >= t->n_tokens) out_of_range++;

    int same = strcmp(back, text) == 0;
    printf("  %-40s токенов %3d  %s  %s\n", text, n,
           same ? "текст совпал" : "текст НЕ совпал",
           out_of_range ? "!! есть id вне словаря" : "id в границах");
    if (!same) printf("      получилось: «%s»\n", back);
    return (!same || out_of_range) ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: tok_roundtrip model.gguf\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "не открыть %s\n", argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "не прочитан\n"); return 2; }
    fclose(f);

    char err[512] = {0};
    gguf_file g;
    if (gguf_open(&g, buf, (size_t)sz, err, sizeof err) != 0) {
        fprintf(stderr, "GGUF: %s\n", err);
        return 2;
    }
    llm_tok t;
    if (llm_tok_open(&t, &g, err, sizeof err) != 0) {
        fprintf(stderr, "tok: %s\n", err);
        return 2;
    }
    printf("словарь: %zu токенов, слияний %zu, стиль %s\n\n", t.n_tokens, t.n_merges,
           t.style_gpt2 ? "gpt2" : "llama");

    int bad = 0;
    bad += check(&t, "The capital of France is");
    bad += check(&t, "hello world");
    bad += check(&t, "Hello, world!");
    bad += check(&t, "don't stop");
    bad += check(&t, "1234567890");
    bad += check(&t, "The quick brown fox jumps over the lazy dog.");
    bad += check(&t, "def f(x):\n    return x * 2\n");
    bad += check(&t, "Привет, мир");
    bad += check(&t, "a  b\tc");

    llm_tok_close(&t);
    gguf_close(&g);
    free(buf);
    printf("\n%s\n", bad ? "есть расхождения" : "все проверки прошли");
    return bad ? 1 : 0;
}