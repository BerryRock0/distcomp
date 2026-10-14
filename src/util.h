#ifndef DC_UTIL_H
#define DC_UTIL_H

#include <stddef.h>

void *xmalloc(size_t n);
void *xcalloc(size_t n, size_t size);
void *xrealloc(void *p, size_t n);
char *xstrdup(const char *s);
char *xstrndup(const char *s, size_t n);

typedef struct {
    char  *p;
    size_t len;
    size_t cap;
} strbuf;

void sb_init(strbuf *b);
void sb_free(strbuf *b);
void sb_addn(strbuf *b, const char *s, size_t n);
void sb_add(strbuf *b, const char *s);
void sb_addc(strbuf *b, char c);
void sb_printf(strbuf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void sb_add_json_string(strbuf *b, const char *s);

extern int g_verbose;

void log_info(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void log_error(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void log_debug(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

char *str_trim(char *s);
int   str_ieq(const char *a, const char *b);
void  str_lower(char *s);
long  now_ms(void);

/* Base64 для двоичных данных, которые едут в JSON: веса модели и
 * активации между слоями. Возвращает длину результата или -1,
 * если в out не влезает (out должен быть на 4/3 от n плюс терминатор).
 * b64_encode_size(n) — сколько байт нужно под результат. */
#define b64_encode_size(n) (((n) + 2) / 3 * 4 + 1)
int b64_encode(const void *in, size_t n, char *out, size_t outsz);
int b64_decode(const char *in, void *out, size_t outsz, size_t *out_n);

#endif /* DC_UTIL_H */
