#include "util.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int g_verbose = 0;

void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fputs("fatal: out of memory\n", stderr);
        abort();
    }
    return p;
}

void *xcalloc(size_t n, size_t size)
{
    void *p = calloc(n ? n : 1, size ? size : 1);
    if (!p) {
        fputs("fatal: out of memory\n", stderr);
        abort();
    }
    return p;
}

void *xrealloc(void *p, size_t n)
{
    void *q = realloc(p, n ? n : 1);
    if (!q) {
        fputs("fatal: out of memory\n", stderr);
        abort();
    }
    return q;
}

char *xstrdup(const char *s)
{
    if (!s) return NULL;
    size_t n = strlen(s);
    char  *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

char *xstrndup(const char *s, size_t n)
{
    if (!s) return NULL;
    size_t len = strnlen(s, n);
    char  *p = xmalloc(len + 1);
    memcpy(p, s, len);
    p[len] = '\0';
    return p;
}

void sb_init(strbuf *b)
{
    b->p = NULL;
    b->len = 0;
    b->cap = 0;
}

void sb_free(strbuf *b)
{
    free(b->p);
    b->p = NULL;
    b->len = b->cap = 0;
}

static void sb_grow(strbuf *b, size_t extra)
{
    if (b->len + extra + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 128;
    while (cap < b->len + extra + 1) cap *= 2;
    b->p = xrealloc(b->p, cap);
    b->cap = cap;
}

void sb_addn(strbuf *b, const char *s, size_t n)
{
    if (!s || !n) return;
    sb_grow(b, n);
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}

void sb_add(strbuf *b, const char *s)
{
    if (s) sb_addn(b, s, strlen(s));
}

void sb_addc(strbuf *b, char c)
{
    sb_grow(b, 1);
    b->p[b->len++] = c;
    b->p[b->len] = '\0';
}

void sb_printf(strbuf *b, const char *fmt, ...)
{
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need < 0) {
        va_end(ap2);
        return;
    }
    sb_grow(b, (size_t)need);
    vsnprintf(b->p + b->len, (size_t)need + 1, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)need;
}

void sb_add_json_string(strbuf *b, const char *s)
{
    sb_addc(b, '"');
    if (s) {
        for (const unsigned char *q = (const unsigned char *)s; *q; q++) {
            switch (*q) {
            case '"':  sb_add(b, "\\\""); break;
            case '\\': sb_add(b, "\\\\"); break;
            case '\n': sb_add(b, "\\n");  break;
            case '\r': sb_add(b, "\\r");  break;
            case '\t': sb_add(b, "\\t");  break;
            case '\b': sb_add(b, "\\b");  break;
            case '\f': sb_add(b, "\\f");  break;
            default:
                if (*q < 0x20) sb_printf(b, "\\u%04x", *q);
                else           sb_addc(b, (char)*q);
            }
        }
    }
    sb_addc(b, '"');
}

static void vlog(const char *level, const char *fmt, va_list ap)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);
    char stamp[32];
    strftime(stamp, sizeof stamp, "%H:%M:%S", &tm);
    fprintf(stderr, "%s.%03ld %-5s ", stamp, ts.tv_nsec / 1000000L, level);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
}

void log_info(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vlog("info", fmt, ap);
    va_end(ap);
}

void log_error(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vlog("ERR", fmt, ap);
    va_end(ap);
}

void log_debug(const char *fmt, ...)
{
    if (!g_verbose) return;
    va_list ap;
    va_start(ap, fmt);
    vlog("dbg", fmt, ap);
    va_end(ap);
}

char *str_trim(char *s)
{
    if (!s) return NULL;
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return s;
}

int str_ieq(const char *a, const char *b)
{
    if (!a || !b) return 0;
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++;
        b++;
    }
    return *a == *b;
}

void str_lower(char *s)
{
    for (; s && *s; s++) *s = (char)tolower((unsigned char)*s);
}

long now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

/* ------------------------------------------------------------- base64 --- */
/* Таблица и разбор написаны без malloc и без sscanf: ими пользуется и узел,
 * где лишняя аллокация стоит дороже самого преобразования. */

static const char b64_tab[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int b64_encode(const void *in, size_t n, char *out, size_t outsz)
{
    const unsigned char *p = in;
    size_t               need = b64_encode_size(n);
    if (!out || outsz < need) return -1;

    size_t o = 0;
    size_t i = 0;
    for (; i + 3 <= n; i += 3) {
        unsigned v = ((unsigned)p[i] << 16) | ((unsigned)p[i + 1] << 8) | p[i + 2];
        out[o++] = b64_tab[(v >> 18) & 63];
        out[o++] = b64_tab[(v >> 12) & 63];
        out[o++] = b64_tab[(v >> 6) & 63];
        out[o++] = b64_tab[v & 63];
    }
    if (i < n) {
        unsigned v = (unsigned)p[i] << 16;
        if (i + 1 < n) v |= (unsigned)p[i + 1] << 8;
        out[o++] = b64_tab[(v >> 18) & 63];
        out[o++] = b64_tab[(v >> 12) & 63];
        out[o++] = (i + 1 < n) ? b64_tab[(v >> 6) & 63] : '=';
        out[o++] = '=';
    }
    out[o] = '\0';
    return (int)o;
}

static int b64_val(char c)
{
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

int b64_decode(const char *in, void *out, size_t outsz, size_t *out_n)
{
    unsigned char *p = out;
    size_t         o = 0;
    unsigned       acc = 0;
    int            bits = 0;

    if (!in) return -1;
    for (const char *q = in; *q; q++) {
        if (*q == '=' || *q == '\n' || *q == '\r' || *q == ' ') continue;
        int v = b64_val(*q);
        if (v < 0) return -1;
        acc = (acc << 6) | (unsigned)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            if (o >= outsz) return -1;
            p[o++] = (unsigned char)((acc >> bits) & 0xFF);
        }
    }
    if (out_n) *out_n = o;
    return 0;
}
