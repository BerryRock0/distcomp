#include "tok.h"
#include "util.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Токенизатор GGUF: словарь и слияния лежат в метаданных файла.
 *
 * BPE (стиль «gpt2») считаем по слияниям: кусок начинается из одиночных
 * символов, затем сливаются пары с наивысшим приоритетом. Приоритет — номер
 * в списке merges: раньше в списке — раньше применяется.
 *
 * Важное: символ здесь не байт, а кодовая точка. Байты текста сначала
 * прогоняются через tables bytes_to_unicode() из GPT-2, и уже эта строка
 * ищется в словаре. Раньше байт подставлялся в токен напрямую, из-за чего
 * получались идентификаторы мимо словаря (промпт превращался в «TArA»,
 * а последний токен вылезал за пределы n_vocab).
 *
 * Пре-токенизация повторяет то, что делает llama.cpp для Qwen2:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   |[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * Регулярных выражений в libc нет, поэтому разбор написан руками — зато
 * без зависимостей.
 */

#define ERR(...)                                                              \
    do {                                                                      \
        snprintf(err, errlen, __VA_ARGS__);                                   \
        return -1;                                                            \
    } while (0)

/* ------------------------------------------------------- таблицы ----- */

/* FNV-1a по строке */
static uint64_t hash_str(const char *s)
{
    uint64_t h = 1469598103934665603ull;
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= 1099511628211ull;
    }
    return h ? h : 1;
}

static int map_put(llm_tok *t, const char *k, int v)
{
    size_t mask = t->map_cap - 1;
    size_t i = (size_t)hash_str(k) & mask;
    while (t->map_key[i]) {
        if (strcmp(t->map_key[i], k) == 0) return 0;   /* первый побеждает */
        i = (i + 1) & mask;
    }
    t->map_key[i] = k;
    t->map_val[i] = v;
    return 0;
}

static int map_get(const llm_tok *t, const char *k)
{
    size_t mask = t->map_cap - 1;
    size_t i = (size_t)hash_str(k) & mask;
    while (t->map_key[i]) {
        if (strcmp(t->map_key[i], k) == 0) return t->map_val[i];
        i = (i + 1) & mask;
    }
    return -1;
}

static int64_t hash_pair(int a, int b)
{
    return (int64_t)(((uint64_t)(uint32_t)a << 32) | (uint32_t)b);
}

static void merge_init(llm_tok *t, size_t n_merges)
{
    size_t cap = 16;
    while (cap < n_merges * 2) cap *= 2;
    t->mg_key = xmalloc(cap * sizeof *t->mg_key);
    t->mg_rank = xmalloc(cap * sizeof *t->mg_rank);
    t->mg_id = xmalloc(cap * sizeof *t->mg_id);
    for (size_t i = 0; i < cap; i++) t->mg_key[i] = -1;
    t->mg_cap = cap;
}

/* Кладёт пару (a, b) с приоритетом rank и id склеенного токена.
 * Именно готовый id, а не n_tokens+rank: слитый токен лежит в словаре
 * по обычным правилам, и вычислять его номер из номера слияния нельзя. */
static void merge_put(llm_tok *t, int a, int b, int rank, int id)
{
    size_t mask = t->mg_cap - 1;
    size_t i = (size_t)hash_pair(a, b) & mask;
    while (t->mg_key[i] >= 0) {
        if (t->mg_key[i] == hash_pair(a, b)) return;   /* первый ранг важнее */
        i = (i + 1) & mask;
    }
    t->mg_key[i] = hash_pair(a, b);
    t->mg_rank[i] = rank;
    t->mg_id[i] = id;
}

/* Приоритет пары; -1 — такой пары нет. */
static int merge_get(const llm_tok *t, int a, int b)
{
    size_t mask = t->mg_cap - 1;
    size_t i = (size_t)hash_pair(a, b) & mask;
    while (t->mg_key[i] >= 0) {
        if (t->mg_key[i] == hash_pair(a, b)) return t->mg_rank[i];
        i = (i + 1) & mask;
    }
    return -1;
}

/* Идентификатор склеенного токена для пары с лучшим рангом. */
static int merge_id(const llm_tok *t, int a, int b)
{
    size_t mask = t->mg_cap - 1;
    size_t i = (size_t)hash_pair(a, b) & mask;
    while (t->mg_key[i] >= 0) {
        if (t->mg_key[i] == hash_pair(a, b)) return t->mg_id[i];
        i = (i + 1) & mask;
    }
    return -1;
}

/* bytes_to_unicode(): байт 0..255 -> кодовая точка. Печатные ASCII и
 * Latin-1 остаются как есть, остальные уезжают в U+0100 и выше. */
static unsigned b2u[256];

static void bytes_to_unicode_init(void)
{
    static int done = 0;
    if (done) return;
    unsigned char used[256] = {0};
    for (int i = '!'; i <= '~'; i++) used[i] = 1;
    for (int i = 0xA1; i <= 0xAC; i++) used[i] = 1;
    for (int i = 0xAE; i <= 0xFF; i++) used[i] = 1;
    unsigned n = 0;
    for (int i = 0; i < 256; i++)
        b2u[i] = used[i] ? (unsigned)i : (unsigned)(256 + n++);
    done = 1;
}

/* кодовая точка -> её UTF-8 запись; возвращает длину */
static int cp_utf8(unsigned cp, char *out)
{
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

/* ------------------------------------------------------- открытие ----- */

int llm_tok_open(llm_tok *t, const gguf_file *g, char *err, size_t errlen)
{
    memset(t, 0, sizeof *t);
    bytes_to_unicode_init();

    const gguf_meta *mt = gguf_find(g, "tokenizer.ggml.tokens");
    if (!mt || mt->type != 9 || mt->arr_type != 8 || mt->arr_len == 0)
        ERR("в модели нет tokenizer.ggml.tokens — токенизатор не собрать");
    t->n_tokens = mt->arr_len;
    t->tokens = xmalloc((t->n_tokens + 1) * sizeof *t->tokens);
    for (size_t i = 0; i < t->n_tokens; i++) t->tokens[i] = mt->arr_str[i];
    t->tokens[t->n_tokens] = NULL;

    const char *model = gguf_str(g, "tokenizer.ggml.model");
    t->style_gpt2 = (model && strcmp(model, "gpt2") == 0);

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

    /* хеш строки -> id: словарь ищется за O(длина), а не перебором */
    size_t cap = 16;
    while (cap < t->n_tokens * 2) cap *= 2;
    t->map_cap = cap;
    t->map_key = xcalloc(cap, sizeof *t->map_key);
    t->map_val = xmalloc(cap * sizeof *t->map_val);
    for (size_t i = 0; i < t->n_tokens; i++) map_put(t, t->tokens[i], (int)i);

    /* слияния: обе половины ищем в словаре, результат — тоже в словаре.
 * Идентификатор склеенного токена не выводится из номера слияния: он
 * лежит в tokens по тем же правилам, что и все остальные. */
    merge_init(t, t->n_merges ? t->n_merges : 1);
    char joined[1024];
    for (size_t i = 0; i < t->n_merges; i++) {
        const char *sp = strchr(t->merges[i], ' ');
        if (!sp) continue;
        char a[512];
        size_t la = (size_t)(sp - t->merges[i]);
        if (la >= sizeof a) continue;
        memcpy(a, t->merges[i], la);
        a[la] = '\0';
        int ia = map_get(t, a);
        int ib = map_get(t, sp + 1);
        if (ia < 0 || ib < 0) continue;
        size_t lj = la + strlen(sp + 1);
        if (lj >= sizeof joined) continue;
        memcpy(joined, a, la);
        memcpy(joined + la, sp + 1, strlen(sp + 1));
        joined[lj] = '\0';
        int ij = map_get(t, joined);
        if (ij >= 0) merge_put(t, ia, ib, (int)i, ij);
    }

    /* Обратное отображение байт -> токен для byte-fallback. */
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
    free(t->map_key);
    free(t->map_val);
    free(t->mg_key);
    free(t->mg_rank);
    free(t->mg_id);
    memset(t, 0, sizeof *t);
}

/* ------------------------------------------------- пре-токенизация ----- */

static int is_letter(unsigned char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 128;
}

static int is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
static int is_nl(unsigned char c)    { return c == '\r' || c == '\n'; }
static int is_ws(unsigned char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

/* «не буква, не цифра, не перевод строки» — класс, который regex называет
 * [^\r\n\p{L}\p{N}] */
static int is_odd(unsigned char c) { return !is_letter(c) && !is_digit(c) && !is_nl(c); }

/* сокращения: 's 't 're 've 'm 'll 'd, регистр не важен */
static int match_contraction(const char *p, int *len)
{
    static const char *sufs[] = { "re", "ve", "ll", "s", "t", "m", "d" };
    if (*p != '\'') return 0;
    for (size_t i = 0; i < sizeof sufs / sizeof sufs[0]; i++) {
        const char *s = sufs[i];
        size_t n = strlen(s);
        if (!strncasecmp(p + 1, s, n)) { *len = (int)n + 1; return 1; }
    }
    return 0;
}

/* Разбивает текст на куски правилом Qwen2. Возвращает число кусков. */
static int pretokenize(const char *text, char **out, int max_out)
{
    int n = 0;
    const char *p = text;
    while (*p && n < max_out) {
        const char *start = p;
        int clen = 0;

        if (match_contraction(p, &clen)) {
            p += clen;
        } else if (is_odd((unsigned char)*p) && p[1] && is_letter((unsigned char)p[1])) {
            /* [^\r\n\p{L}\p{N}]?\p{L}+ — один «странный» символ и буквы */
            p++;
            while (*p && is_letter((unsigned char)*p)) p++;
        } else if (is_letter((unsigned char)*p)) {
            while (*p && is_letter((unsigned char)*p)) p++;
        } else if (is_digit((unsigned char)*p)) {
            /* \p{N}{1,3} — цифры группами не длиннее трёх */
            int k = 0;
            while (*p && is_digit((unsigned char)*p) && k < 3) { p++; k++; }
        } else if ((*p == ' ' || (*p && is_odd((unsigned char)*p) &&
                     !is_ws((unsigned char)*p) && p[1] &&
                     !is_ws((unsigned char)p[1]))) &&
                   /*  ?[^\s\p{L}\p{N}]+ */
                   ((*p == ' ' && p[1] && !is_ws((unsigned char)p[1]) &&
                     !is_letter((unsigned char)p[1]) && !is_digit((unsigned char)p[1])) ||
                    (*p != ' ' && is_odd((unsigned char)*p)))) {
            if (*p == ' ') p++;
            while (*p && !is_ws((unsigned char)*p) && !is_letter((unsigned char)*p) &&
                   !is_digit((unsigned char)*p))
                p++;
            while (*p && is_nl((unsigned char)*p)) p++;   /* [\r\n]* */
        } else if (is_ws((unsigned char)*p)) {
            /* \s*[\r\n]+ — пробельная серия, если она кончается переводом */
            const char *q = p;
            const char *last_nl = NULL;
            while (*q && is_ws((unsigned char)*q)) {
                if (is_nl((unsigned char)*q)) last_nl = q;
                q++;
            }
            if (last_nl) p = last_nl + 1;
            else          p = q;
        } else {
            p++;   /* запасной путь: один символ */
        }

        if (p == start) p++;
        out[n++] = xstrndup(start, (size_t)(p - start));
    }
    return n;
}

/* --------------------------------------------------- BPE одного куска --- */

/* Символы куска (байты исходника) сначала становятся кодовыми точками
 * bytes_to_unicode, потом ищутся в словаре. */
static int bpe_encode(const llm_tok *t, const char *text, int *out, int max_out,
                      int *tmp, int max_tmp, char *err, size_t errlen)
{
    int n = 0;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (n >= max_tmp) break;
        char cp[4];
        int  l = cp_utf8(b2u[*p], cp);
        cp[l] = '\0';
        int id = map_get(t, cp);
        if (id < 0) {
            /* Такого символа в словаре нет — подставлять «сырой байт как
             * токен» нельзя: получится id мимо словаря. */
            snprintf(err, errlen,
                     "в словаре нет символа для байта 0x%02X (токен не собран)", *p);
            return -1;
        }
        tmp[n++] = id;
    }

    for (;;) {
        int best_rank = -1, best_i = -1;
        for (int i = 0; i + 1 < n; i++) {
            int r = merge_get(t, tmp[i], tmp[i + 1]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) { best_rank = r; best_i = i; }
        }
        if (best_i < 0) break;
        tmp[best_i] = merge_id(t, tmp[best_i], tmp[best_i + 1]);
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
        char *pieces[2048];
        int np = pretokenize(text, pieces, (int)(sizeof pieces / sizeof pieces[0]));
        if (np >= (int)(sizeof pieces / sizeof pieces[0]))
            np = (int)(sizeof pieces / sizeof pieces[0]) - 1;
        int tmp[1024];
        for (int i = 0; i < np && ntok < max_out; i++) {
            int got = bpe_encode(t, pieces[i], out + ntok, max_out - ntok, tmp,
                                 (int)(sizeof tmp / sizeof tmp[0]), err, errlen);
            free(pieces[i]);
            if (got < 0) return -1;
            ntok += got;
        }
    } else {
        /* стиль llama: пробел -> ▁, дальше жадный поиск по словарю */
        strbuf b;
        sb_init(&b);
        for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
            if (*p == ' ') sb_add(&b, "\xe2\x96\x81");
            else sb_addc(&b, (char)*p);
        }
        size_t len = b.len;
        char  *buf = b.p;
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
                unsigned char c = buf[i];
                if (t->byte_tok[c]) out[ntok++] = t->byte_tok[c] - 1;
                else                out[ntok++] = (int)c;
                i++;
            }
        }
        sb_free(&b);
    }

    if (t->add_eos && t->eos_id > 0 && ntok < max_out) out[ntok++] = t->eos_id;
    return ntok;
}

/* Кодовая точка из строки токена -> байт обратно (для byte-level). */
static int cp_to_byte(unsigned cp, unsigned char *out)
{
    if (cp < 256) { *out = (unsigned char)cp; return 1; }
    /* ищем байт, который в bytes_to_unicode даёт именно эту точку */
    for (int b = 0; b < 256; b++)
        if (b2u[b] == cp) { *out = (unsigned char)b; return 1; }
    return 0;
}

int llm_tok_decode(const llm_tok *t, const int *ids, int n, char *out, size_t outsz)
{
    if (!t || !out || outsz == 0) return -1;
    size_t used = 0;
    out[0] = '\0';

    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || (size_t)ids[i] >= t->n_tokens) continue;
        const char *s = t->tokens[ids[i]];
        size_t      l = strlen(s);

        /* плейсхолдер «<0xNN>» — это один байт, а не шесть символов */
        if (l == 6 && memcmp(s, "<0x", 3) == 0) {
            int v = (int)strtol(s + 3, NULL, 16);
            if (v >= 0 && v < 256 && used + 1 < outsz) out[used++] = (char)v;
            continue;
        }

        if (t->style_gpt2) {
            /* кодовые точки обратно в байты исходного текста */
            size_t k = 0;
            while (k < l) {
                unsigned char c = (unsigned char)s[k];
                unsigned cp;
                int extra;
                if (c < 0x80)         { cp = c;        extra = 0; }
                else if (c < 0xE0)    { cp = c & 0x1F; extra = 1; }
                else if (c < 0xF0)    { cp = c & 0x0F; extra = 2; }
                else                  { cp = c & 0x07; extra = 3; }
                k++;
                for (int q = 0; q < extra && k < l; q++, k++)
                    cp = (cp << 6) | ((unsigned char)s[k] & 0x3F);
                unsigned char b;
                if (cp_to_byte(cp, &b) && used + 1 < outsz) out[used++] = (char)b;
            }
            continue;
        }

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