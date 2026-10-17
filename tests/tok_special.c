/* Проверка того, что служебные токены вида <|im_start|> кодируются как есть.
 *
 * Через BPE такой текст распадается на обычные куски («<», «|», «im», …), и
 * модель не видит разделителей реплик: она продолжает ответ за пределы своей
 * реплики и разговаривает сама с собой.
 *
 *   tok_special <model.gguf>
 */
#include "gguf.h"
#include "tok.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void show(const llm_tok *t, const char *label, const char *text)
{
    int ids[256];
    int n = llm_tok_encode(t, text, ids, 256, NULL, 0);
    printf("%-22s %2d токенов:", label, n);
    for (int i = 0; i < n; i++) printf(" %d", ids[i]);
    printf("\n");
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "model.gguf";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "нет файла %s\n", path); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "короткий файл\n"); return 2; }
    fclose(f);

    char err[512] = {0};
    gguf_file g;
    if (gguf_open(&g, buf, (size_t)sz, err, sizeof err) != 0) {
        fprintf(stderr, "GGUF: %s\n", err); return 2;
    }
    llm_tok t;
    if (llm_tok_open(&t, &g, err, sizeof err) != 0) {
        fprintf(stderr, "токенизатор: %s\n", err); return 2;
    }
    printf("служебных токенов в словаре: %zu\n", t.spec_n);
    for (size_t i = 0; i < t.spec_n && i < 8; i++)
        printf("  %s -> %d\n", t.spec_str[i], t.spec_id[i]);
    if (!t.spec_n) { printf("нет служебных токенов вида <|…|>\n"); return 0; }

    /* эталонные id: <|im_start|> и <|im_end|> должны кодироваться в один токен */
    int im_start = -1, im_end = -1;
    for (size_t i = 0; i < t.spec_n; i++) {
        if (!strcmp(t.spec_str[i], "<|im_start|>")) im_start = t.spec_id[i];
        if (!strcmp(t.spec_str[i], "<|im_end|>"))   im_end   = t.spec_id[i];
    }
    printf("\nim_start=%d im_end=%d\n", im_start, im_end);

    int bad = 0;
    int ids[256];
    int n = llm_tok_encode(&t, "<|im_start|>привет<|im_end|>", ids, 256, NULL, 0);
    printf("код: ");
    for (int i = 0; i < n; i++) printf("%d ", ids[i]);
    printf("\n");
    if (im_start >= 0 && ids[0] != im_start) {
        printf("ОШИБКА: первый токен %d, а должен быть <|im_start|> = %d\n", ids[0], im_start);
        bad++;
    }
    if (im_end >= 0 && ids[n - 1] != im_end) {
        printf("ОШИБКА: последний токен %d, а должен быть <|im_end|> = %d\n", ids[n - 1], im_end);
        bad++;
    }
    /* «привет» сам по себе занимает несколько BPE-токенов, поэтому сверяем не
     * число, а состав: служебные по краям, а между ними ровно то же, что даёт
     * кодирование одного только слова */
    int mid[256];
    int nm = llm_tok_encode(&t, "привет", mid, 256, NULL, 0);
    if (n != nm + 2) {
        printf("ОШИБКА: ожидалось %d токенов (start + %d + end), получено %d\n",
               nm + 2, nm, n);
        bad++;
    } else {
        int same = 1;
        for (int i = 0; i < nm; i++) if (ids[i + 1] != mid[i]) same = 0;
        if (!same) { printf("ОШИБКА: середина не совпала с кодированием слова\n"); bad++; }
    }

    show(&t, "обычный текст", "привет");
    printf("\n%s\n", bad ? "есть ошибки" : "служебные токены кодируются верно");
    return bad ? 1 : 0;
}
