/* Проверка деквантизации на настоящем файле: раскрываем несколько
 * квантовых тензоров и смотрим, похожи ли числа на веса нейросети
 * (нет NaN/Inf, среднее около нуля, разброс правдоподобный). */
#include "gguf.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc < 2) { printf("нужен путь к .gguf\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { printf("не открыть %s\n", argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { printf("не прочитан\n"); return 2; }
    fclose(f);

    gguf_file g;
    char err[512] = {0};
    if (gguf_open(&g, buf, (size_t)sz, err, sizeof err) != 0) {
        printf("GGUF: %s\n", err);
        return 2;
    }
    printf("архитектура: %s, тензоров %zu\n\n", gguf_str(&g, "general.architecture"), g.n_tensors);

    printf("%-28s %-6s %10s %10s %9s %8s %6s\n",
           "тензор", "тип", "min", "max", "среднее", "сигма", "плохих");
    int bad_total = 0;
    for (size_t i = 0; i < g.n_tensors && i < 6; i++) {
        const gguf_tensor *t = &g.tensors[i];
        size_t nelem = 1;
        for (unsigned d = 0; d < t->n_dims; d++) nelem *= (size_t)t->dims[d];
        if (nelem > 262144) nelem = 262144;   /* хватит выборки */

        float *v = xmalloc(nelem * sizeof *v);
        if (gguf_dequant(t, v, 0, nelem) != 0) {
            printf("%-28s %-6u  НЕ РАСКРЫТ\n", t->name, t->type);
            free(v);
            bad_total++;
            continue;
        }
        double sum = 0, sq = 0, mn = 1e30, mx = -1e30;
        int bad = 0;
        for (size_t k = 0; k < nelem; k++) {
            double d = v[k];
            if (!isfinite(d)) { bad++; continue; }
            sum += d;
            sq  += d * d;
            if (d < mn) mn = d;
            if (d > mx) mx = d;
        }
        double mean = sum / (double)nelem;
        double var  = sq / (double)nelem - mean * mean;
        printf("%-28s %-6u %10.4f %10.4f %9.5f %8.4f %6d\n", t->name, t->type, mn, mx, mean,
               sqrt(var > 0 ? var : 0), bad);
        bad_total += bad;
        free(v);
    }
    gguf_close(&g);
    free(buf);
    printf("\nитого нечисловых значений: %d\n", bad_total);
    return bad_total ? 1 : 0;
}