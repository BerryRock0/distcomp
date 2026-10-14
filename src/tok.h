#ifndef DC_TOK_H
#define DC_TOK_H

#include "gguf.h"

#include <stddef.h>

/*
 * Токенизатор для моделей llama-семейства.
 *
 * GGUF несёт словарь прямо в метаданных: tokenizer.ggml.tokens (строки),
 * tokenizer.ggml.merges (пары для BPE) или только tokens (для стиля «llama»,
 * где слияния не нужны — токены уже готовые). Оба варианта собраны здесь,
 * потому что и те, и другие встречаются в реальных файлах.
 *
 * Стиль «gpt2»: словарь начинается с 256 байтовых токенов, дальше слияния.
 * Пре-токенизация — упрощённое правило GPT-2 (слова, числа, пунктуация,
 * пробелы отдельно), полный regex llama.cpp здесь не нужен: на нём словарь
 * и не обучен.
 *
 * Стиль «llama»: токены — куски текста, пробел помечен как ▁ (U+2581).
 * Непечатные байты разворачиваются обратно в байты (byte-fallback).
 */

typedef struct {
    char   **tokens;
    size_t   n_tokens;
    char   **merges;          /* пары через пробел */
    size_t   n_merges;
    int      style_gpt2;      /* 1 — BPE по слияниям, 0 — стиль llama */
    int      add_bos, add_eos;
    int      bos_id, eos_id;
    unsigned char byte_tok[256];   /* обратное отображение байт -> токен */
} llm_tok;

int  llm_tok_open(llm_tok *t, const gguf_file *g, char *err, size_t errlen);
void llm_tok_close(llm_tok *t);

/* Текст -> идентификаторы. Возвращает число токенов или -1.
 * out должен вмещать max_out элементов. */
int  llm_tok_encode(const llm_tok *t, const char *text, int *out, int max_out,
                   char *err, size_t errlen);

/* Идентификаторы -> текст (модель «написала» это). */
int  llm_tok_decode(const llm_tok *t, const int *ids, int n, char *out, size_t outsz);

#endif /* DC_TOK_H */