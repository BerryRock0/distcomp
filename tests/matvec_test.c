/* Проверка того, какую раскладку матрицы считает matvec.
 *
 * В файле ggml тензор [ne0, ne1] лежит так: элемент (i0, i1) по адресу
 * i0 + i1*ne0, то есть непрерывное измерение — ne0 (вход). Тест кладёт в
 * data известные значения в этой раскладке и сравнивает результат matvec
 * с математически верным y[out] = sum_in W[in + out*ne0] * x[in].
 *
 * Если движок читает транспонированно, значения сойдутся только с
 * переставленными n_in/n_out — и это надо знать до внедрения квантов. */
#include "engine.c"
#include "util.h"

#include <math.h>
#include <stdio.h>

static int close_enough(float a, float b)
{
    return fabsf(a - b) < 1e-4f;
}

int main(void)
{
    const int NE0 = 4;   /* ne0 — вход (непрерывное измерение) */
    const int NE1 = 3;   /* ne1 — выход */
    const size_t N = (size_t)NE0 * NE1;

    /* W[in + out*NE0] = in * 10 + out — все значения различимы */
    float ref[16];
    for (size_t out = 0; out < NE1; out++)
        for (size_t in = 0; in < NE0; in++)
            ref[in + out * NE0] = (float)(in * 10 + out) / 100.0f;

    gguf_tensor t;
    memset(&t, 0, sizeof t);
    t.type = 0;              /* F32 */
    t.n_dims = 2;
    t.dims[0] = NE0;
    t.dims[1] = NE1;
    t.data = (const unsigned char *)ref;

    float x[8], scratch[8], y[8];
    for (int i = 0; i < NE0; i++) x[i] = (float)(i + 1);

    /* Правильный результат: y[out] = sum_in W[in + out*NE0] * x[in] */
    float want[8];
    for (int out = 0; out < NE1; out++) {
        double s = 0;
        for (int in = 0; in < NE0; in++) s += (double)ref[in + out * NE0] * x[in];
        want[out] = (float)s;
    }

    /* Вызов ровно как в движке: n_in = ne0 (вход), n_out = ne1 (выход) */
    matvec(y, &t, x, NE0, NE1, scratch);

    printf("n_in=%d n_out=%d\n", NE0, NE1);
    printf("ожидалось : ");
    for (int j = 0; j < NE1; j++) printf("%8.4f", want[j]);
    printf("\nполучено  : ");
    for (int j = 0; j < NE1; j++) printf("%8.4f", y[j]);
    printf("\n");

    int ok = 1;
    for (int j = 0; j < NE1; j++)
        if (!close_enough(y[j], want[j])) ok = 0;
    printf("%s\n", ok ? "РАСКЛАДКА СОВПАДАЕТ: matvec читает как ggml"
                       : "РАСХОЖДЕНИЕ: matvec читает транспонированно");
    return ok ? 0 : 1;
}