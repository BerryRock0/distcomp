#ifndef DC_TOK_H
#define DC_TOK_H

#include "gguf.h"

#include <stddef.h>
#include <stdint.h>

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

    /* Строка -> id и пара -> ранг. Собраны один раз при открытии: раньше
     * слияния искались перебором всего словаря (151387 слияний на 151936
     * токенов — это десятки миллиардов сравнений), а это занимало минуты
     * на каждый промпт. */
    const char **map_key;
    int         *map_val;
    size_t       map_cap;
    int64_t     *mg_key;      /* hash_pair(a,b), -1 — пусто */
    int         *mg_rank;     /* приоритет: позиция в списке слияний */
    int         *mg_id;       /* идентификатор склеенного токена */
    size_t       mg_cap;

    /* Служебные токены вида <|im_start|>, <|im_end|>. Обычный BPE их не
     * различает: он режет такой текст на пять обычных кусков, и модель
     * никогда не видит настоящий разделитель реплик. Поэтому они вылавливаются
     * из текста напрямую. Список выбирается из самого словаря — у Qwen2,
     * Llama-3 и Gemma таких токенов по несколько. */
    const char **spec_str;
    int         *spec_id;
    size_t       spec_n;
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