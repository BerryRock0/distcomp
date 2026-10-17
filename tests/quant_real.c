/*
 * Сверка деквантизации на настоящих блоках из модели: все 256 весов.
 *
 * Эталон снимает scripts/dump_blocks.py (пакет gguf, порт ggml). Настоящие
 * блоки, а не два синтетических: проверка первых 64 весов пропускала ошибки
 * раскладки в группах 4..7, то есть в весах 128..255.
 *
 *   quant_real blocks.bin
 */
#include "gguf.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *tname(unsigned qt)
{
    switch (qt) {
    case 10: return "Q2_K";
    case 11: return "Q3_K";
    case 12: return "Q4_K";
    case 13: return "Q5_K";
    case 14: return "Q6_K";
    case 15: return "Q8_K";
    default: return "??";
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "нужен blocks.bin\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "не открыть %s\n", argv[1]); return 2; }

    unsigned ntypes = 0;
    if (fread(&ntypes, 4, 1, f) != 1) return 2;

    int bad_total = 0;
    for (unsigned ti = 0; ti < ntypes; ti++) {
        unsigned qt = 0, nb = 0;
        if (fread(&qt, 4, 1, f) != 1) return 2;
        if (fread(&nb, 4, 1, f) != 1) return 2;

        size_t bytes_per = (qt == 10 ? 84 : qt == 11 ? 110 : qt == 12 ? 144 :
                            qt == 13 ? 176 : qt == 14 ? 210 : 292);
        unsigned char *blocks = xmalloc((size_t)nb * bytes_per);
        float       *truth  = xmalloc((size_t)nb * 256 * sizeof(float));
        if (fread(blocks, bytes_per, nb, f) != nb) { fprintf(stderr, "короткий файл\n"); return 2; }
        if (fread(truth, sizeof(float), (size_t)nb * 256, f) != (size_t)nb * 256) {
            fprintf(stderr, "не хватило эталона\n");
            return 2;
        }

        /* один блок = один gguf-тензор минимального размера */
        gguf_tensor t;
        memset(&t, 0, sizeof t);
        t.type  = qt;
        t.n_dims = 1;
        t.dims[0] = (unsigned long)(256 * nb);
        t.data  = blocks;
        t.nbytes = bytes_per * nb;

        float *got = xmalloc((size_t)nb * 256 * sizeof(float));
        if (gguf_dequant(&t, got, 0, (size_t)nb * 256) != 0) {
            printf("  %-6s gguf_dequant вернул ошибку\n", tname(qt));
            bad_total++;
        } else {
            int bad = 0;
            size_t first = 0;
            for (size_t i = 0; i < (size_t)nb * 256; i++) {
                float a = got[i], b = truth[i];
                if (!(fabsf(a - b) <= 1e-4f * (fabsf(b) + 1e-3f))) {
                    if (!bad) first = i;
                    bad++;
                }
            }
            printf("  %-6s блоков %3u: %s", tname(qt), nb,
                   bad ? "расходится" : "совпадает с эталоном");
            if (bad) {
                printf(" (%d из %zu, первый вес %zu: %.6f против %.6f)\n",
                       bad, (size_t)nb * 256, first, got[first], truth[first]);
            } else {
                putchar('\n');
            }
            bad_total += bad;
        }
        free(blocks); free(truth); free(got);
    }
    fclose(f);
    printf("\n%s\n", bad_total ? "есть расхождения с эталоном" : "все типы совпали с эталоном");
    return bad_total ? 1 : 0;
}
