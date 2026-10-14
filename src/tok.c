#include "tok.h"
#include "util.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Токенизатор GGUF: словарь и слияния лежат в метаданных файла.
 *
 * BPE (стиль «gpt2») считаем по слияниям: слово начинается из одиночных
 * символов, затем сливаются пары с наивысшим приоритетом, пока есть что
 * сливать. Приоритет — номер в списке merges: раньше в списке — раньше
 * применяется.
 */

#define ERR(...)                                                              \
    do {                                                                      \
        snprintf(err, errlen, __VA_ARGS__);                                   \
        return -1;                                                            \
    } while (0)

/* ------------------------------------------------------------ словарь --- */

static int64_t hash_pair(int a, int b)
{
    return (int64_t)((uint64_t)(uint32_t)a << 32 | (uint32_t)b);
}

typedef struct {
    int64_t *key;     /* hash_pair(a, b), -1 — пусто */
    int    *rank;
    size_t  cap;
    size_t  used;
} merge_table;

static void merge_init(merge_table *mt, size_t n_merges)
{
    size_t cap = 16;
    while (cap < n_merges * 2) cap *= 2;
    mt->key = xmalloc(cap * sizeof *mt->key);
    mt->rank = xmalloc(cap * sizeof *mt->rank);
    for (size_t i = 0; i < cap; i++) mt->key[i] = -1;
    mt->cap = cap;
    mt->used = 0;
}

static void merge_free(merge_table *mt)
{
    free(mt->key);
    free(mt->rank);
}

static void merge_put(merge_table *mt, int a, int b, int rank)
{
    size_t mask = mt->cap - 1;
    size_t i = (size_t)hash_pair(a, b) & mask;
    while (mt->key[i] >= 0) {
        if (mt->key[i] == hash_pair(a, b)) return;   /* первый ранг важнее */
        i = (i + 1) & mask;
    }
    mt->key[i] = hash_pair(a, b);
    mt->rank[i] = rank;
    mt->used++;
}

static int merge_get(const merge_table *mt, int a, int b)
{
    size_t mask = mt->cap - 1;
    size_t i = (size_t)hash_pair(a, b) & mask;
    while (mt->key[i] >= 0) {
        if (mt->key[i] == hash_pair(a, b)) return mt->rank[i];
        i = (i + 1) & mask;
    }
    return -1;
}

int llm_tok_open(llm_tok *t, const gguf_file *g, char *err, size_t errlen)
{
    memset(t, 0, sizeof *t);

    const gguf_meta *mt = gguf_find(g, "tokenizer.ggml.tokens");
    if (!mt || mt->type != 9 || mt->arr_type != 8 || mt->arr_len == 0)
        ERR("в модели нет tokenizer.ggml.tokens — токенизатор не собрать");
    t->n_tokens = mt->arr_len;
    t->tokens = xmalloc((t->n_tokens + 1) * sizeof *t->tokens);
    for (size_t i = 0; i < t->n_tokens; i++) t->tokens[i] = mt->arr_str[i];
    t->tokens[t->n_tokens] = NULL;

    const char *model = gguf_str(g, "tokenizer.ggml.model");
    t->style_gpt2 = (model && strcmp(model, "gpt2") == 0);

    /* слияния есть не у всех: у стиля llama токены уже готовые */
    const gguf_meta *mm = gguf_find(g, "tokenizer.ggml.merges");
    if (mm && mm->type == 9 && mm->arr_type == 8) {
        t->n_merges = mm->arr_len;
        t->merges = xmalloc((t->n_merges + 1) * sizeof *t->merges);
        for (size_t i = 0; i < t->n_merges; i++) t->merges[i] = mm->arr_str[i];
        t->merges[t->n_merges] = NULL;
    }

    unsigned long bos = 0, eos = 0;
    if (gguf_u32(g, &bos, "tokenizer.ggml.bos_token_id", NULL) == 0) t->bos_id = (int)bos;
    if (gguf_u32(g, &eos, "tokenizer.ggml.eos_token_id", NULL) == 0) t->eos_id = (int)eos;
    unsigned long ab = 0, ae = 0;
    if (gguf_u32(g, &ab, "tokenizer.ggml.add_bos_token", NULL) == 0) t->add_bos = ab != 0;
    if (gguf_u32(g, &ae, "tokenizer.ggml.add_eos_token", NULL) == 0) t->add_eos = ae != 0;

    /* Обратное отображение байт -> токен. У печатных символов токен совпадает
     * с самим символом; у непечатных в словаре лежит плейсхолдер «<0xNN>»,
     * и его приходится искать поиском. Без этого любой непечатный байт
     * уходил бы в сеть как несколько UTF-8 токенов вместо одного. */
    for (size_t i = 0; i < t->n_tokens; i++) {
        size_t l = strlen(t->tokens[i]);
        if (l == 1) t->byte_tok[(unsigned char)t->tokens[i][0]] = (unsigned char)(i + 1);
        else if (l == 6 && memcmp(t->tokens[i], "<0x", 3) == 0) {
            int v = (int)strtol(t->tokens[i] + 3, NULL, 16);
            if (v >= 0 && v < 256) t->byte_tok[v] = (unsigned char)(i + 1);
        }
    }
    return 0;
}

void llm_tok_close(llm_tok *t)
{
    if (!t) return;
    free(t->tokens);
    free(t->merges);
    memset(t, 0, sizeof *t);
}

/* ------------------------------------------------------- пре-токенизация --- */

typedef struct {
    char *s;
    int   id;
} piece;

/* Буква, цифра или иной символ — три разные группы для BPE. */
static int char_class(int c)
{
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return 0;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 128) return 1;
    if (c >= '0' && c <= '9') return 2;
    return 3;
}

/* Разбивает текст на куски по правилу GPT-2: пробелы принадлежат следующему
 * непробельному куску, буквы/цифры/прочие идут отдельными группами.
 * Возвращает число кусков; каждый кусок — подстрока исходного текста. */
static int split_gpt2(const char *text, char **out, int max_out)
{
    int n = 0;
    const char *p = text;
    while (*p && n < max_out) {
        /* пробелы перед непробельным символом — часть следующего куска */
        const char *start = p;
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (!*p) {
            /* хвост из пробелов — отдельным куском */
            if (p > start) {
                out[n] = xstrndup(start, (size_t)(p - start));
                n++;
            }
            break;
        }
        int cl = char_class((unsigned char)*p);
        if (cl == 0) {
            /* сюда не попадаем: пробелы уже съедены выше */
            out[n++] = xstrndup(p++, 1);
            continue;
        }
        while (*p && char_class((unsigned char)*p) == cl) p++;
        /* объединяем накопленные пробелы с куском */
        out[n] = xstrndup(start, (size_t)(p - start));
        n++;
    }
    return n;
}

/* BPE одного куска: символы -> слияния по приоритету. */
static int bpe_encode(const llm_tok *t, const char *text, int *out, int max_out,
                      merge_table *mt, int *tmp, int max_tmp)
{
    int n = 0;
    for (const char *p = text; *p && n < max_tmp; p++) {
        unsigned char c = (unsigned char)*p;
        if (t->byte_tok[c]) tmp[n++] = t->byte_tok[c] - 1;
        else tmp[n++] = c;      /* запасной путь: сырой байт как токен */
    }

    /* слияния: каждый раз ищем пару с лучшим рангом и сливаем */
    for (;;) {
        int best_rank = -1, best_i = -1;
        for (int i = 0; i + 1 < n; i++) {
            int r = merge_get(mt, tmp[i], tmp[i + 1]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) { best_rank = r; best_i = i; }
        }
        if (best_i < 0) break;
        tmp[best_i] = (int)t->n_tokens + best_rank;   /* токен слияния */
        memmove(tmp + best_i + 1, tmp + best_i + 2, sizeof(int) * (size_t)(n - best_i - 2));
        n--;
        if (n <= 1) break;
    }

    int k = 0;
    for (int i = 0; i < n && k < max_out; i++) out[k++] = tmp[i];
    return k;
}

int llm_tok_encode(const llm_tok *t, const char *text, int *out, int max_out, char *err,
                   size_t errlen)
{
    if (!t || !text || !out || max_out <= 0) return -1;

    int ntok = 0;
    if (t->add_bos && t->bos_id > 0) out[ntok++] = t->bos_id;

    if (t->style_gpt2) {
        char *pieces[4096];
        int np = split_gpt2(text, pieces, (int)(sizeof pieces / sizeof pieces[0]));
        if (np >= (int)(sizeof pieces / sizeof pieces[0])) np = (int)(sizeof pieces / sizeof pieces[0]) - 1;
        merge_table mt;
        merge_init(&mt, t->n_merges);
        for (size_t i = 0; i < t->n_merges; i++) {
            char *sp = strchr(t->merges[i], ' ');
            if (!sp) continue;
            int a = -1, b = -1;
            /* ищем оба слова в словаре: их id и есть пара */
            for (size_t k = 0; k < t->n_tokens; k++) {
                size_t l = strlen(t->tokens[k]);
                if (a < 0 && l == (size_t)(sp - t->merges[i]) &&
                    memcmp(t->tokens[k], t->merges[i], l) == 0) a = (int)k;
                if (b < 0 && strcmp(t->tokens[k], sp + 1) == 0) b = (int)k;
                if (a >= 0 && b >= 0) break;
            }
            if (a >= 0 && b >= 0) merge_put(&mt, a, b, (int)i);
        }
        int tmp[1024];
        for (int i = 0; i < np && ntok < max_out; i++) {
            int got = bpe_encode(t, pieces[i], out + ntok, max_out - ntok, &mt, tmp,
                                 (int)(sizeof tmp / sizeof tmp[0]));
            ntok += got;
            free(pieces[i]);
        }
        merge_free(&mt);
    } else {
        /* стиль llama: пробел -> ▁, дальше жадный поиск по словарю */
        strbuf b;
        sb_init(&b);
        for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
            if (*p == ' ') sb_add(&b, "\xe2\x96\x81");
            else sb_addc(&b, (char)*p);
        }
        size_t len = b.len;
        char *buf = b.p;
        /* жадный поиск самого длинного токена */
        for (size_t i = 0; i < len && ntok < max_out;) {
            int best = -1;
            size_t best_len = 0;
            for (size_t k = 0; k < t->n_tokens; k++) {
                size_t l = strlen(t->tokens[k]);
                if (l > best_len && l <= len - i && memcmp(t->tokens[k], buf + i, l) == 0) {
                    best = (int)k;
                    best_len = l;
                }
            }
            if (best >= 0) {
                out[ntok++] = best;
                i += best_len;
            } else {
                /* byte-fallback: берём байт */
                unsigned char c = buf[i];
                if (t->byte_tok[c]) out[ntok++] = t->byte_tok[c] - 1;
                else out[ntok++] = (int)c;
                i++;
            }
        }
        sb_free(&b);
    }

    if (t->add_eos && t->eos_id > 0 && ntok < max_out) out[ntok++] = t->eos_id;
    return ntok;
}

int llm_tok_decode(const llm_tok *t, const int *ids, int n, char *out, size_t outsz)
{
    if (!t || !out || outsz == 0) return -1;
    size_t used = 0;
    out[0] = '\0';
    for (int i = 0; i < n; i++) {
        const char *s = (ids[i] >= 0 && (size_t)ids[i] < t->n_tokens) ? t->tokens[ids[i]] : "";
        /* плейсхолдер «<0xNN>» — это один байт, а не шесть символов */
        if (strlen(s) == 6 && memcmp(s, "<0x", 3) == 0) {
            int v = (int)strtol(s + 3, NULL, 16);
            if (v >= 0 && v < 256 && used + 1 < outsz) out[used++] = (char)v;
            continue;
        }
        size_t l = strlen(s);
        if (used + l + 1 > outsz) break;
        memcpy(out + used, s, l);
        used += l;
    }
    out[used] = '\0';
    /* стиль llama: ▁ обратно в пробел */
    if (!t->style_gpt2) {
        for (size_t i = 0; i + 2 < used; i++) {
            if ((unsigned char)out[i] == 0xe2 && (unsigned char)out[i + 1] == 0x96 &&
                (unsigned char)out[i + 2] == 0x81) {
                out[i] = ' ';
                memmove(out + i + 1, out + i + 3, used - i - 2);
                used -= 2;
            }
        }
        out[used] = '\0';
    }
    return 0;
}

#undef ERR