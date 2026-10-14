#include "gguf.h"
#include "util.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Значения типов из ggml.h — те же номера, что в файле. */
enum {
    T_UINT8 = 0, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32,
    T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64
};

/* В v2 строки и длины массивов — uint64, в v3 — uint32. Разница ровно в
 * одном месте, но модель, записанная старым llama.cpp, обязана читаться. */
typedef struct {
    const unsigned char *p;
    size_t               size;
    unsigned             version;
    int                  bad;
    char                *err;
    size_t               errlen;
} rd;

static void rfail(rd *r, const char *msg)
{
    if (r->bad) return;
    r->bad = 1;
    if (r->err && r->errlen) snprintf(r->err, r->errlen, "%s", msg);
}

static unsigned r_u8(rd *r)
{
    if (r->p + 1 > r->p + r->size) { rfail(r, "файл оборван"); return 0; }
    return *r->p++;
}

static unsigned r_u16(rd *r)
{
    unsigned a = r_u8(r), b = r_u8(r);
    return a | (b << 8);
}

static unsigned r_u32(rd *r)
{
    unsigned a = r_u16(r), b = r_u16(r);
    return a | (b << 16);
}

static unsigned long r_u64(rd *r)
{
    unsigned long a = r_u32(r), b = r_u32(r);
    return a | (b << 32);
}

/* Длина строки или массива: в v2 это uint64, в v3 — uint32.
 * Счётчики в заголовке файла (число тензоров и ключей) всегда uint64,
 * их читает r_count. */
static size_t r_len(rd *r)
{
    unsigned long v = r->version >= 3 ? r_u32(r) : r_u64(r);
    if (r->bad) return 0;
    if (v > r->size) { rfail(r, "строка длиннее файла"); return 0; }
    return (size_t)v;
}

static size_t r_count(rd *r)
{
    unsigned long v = r_u64(r);
    if (r->bad) return 0;
    if (v > r->size) { rfail(r, "счётчик больше размера файла"); return 0; }
    return (size_t)v;
}

static char *r_str(rd *r)
{
    size_t n = r_len(r);
    if (r->bad) return NULL;
    char *s = xmalloc(n + 1);
    if (r->p + n > r->p + r->size) { rfail(r, "строка обрезана"); free(s); return NULL; }
    memcpy(s, r->p, n);
    s[n] = '\0';
    r->p += n;
    return s;
}

static double r_num(rd *r, unsigned type)
{
    switch (type) {
    case T_UINT8:   return r_u8(r);
    case T_INT8:    return (int)(signed char)r_u8(r);
    case T_UINT16:  return r_u16(r);
    case T_INT16:   return (int)(short)r_u16(r);
    case T_UINT32:  return r_u32(r);
    case T_INT32:   return (int)(int32_t)r_u32(r);
    case T_FLOAT32: { unsigned v = r_u32(r); float f; memcpy(&f, &v, 4); return f; }
    case T_BOOL:    return r_u8(r) ? 1 : 0;
    case T_UINT64:  return (double)r_u64(r);
    case T_INT64:   { unsigned long v = r_u64(r); int64_t s; memcpy(&s, &v, 8); return (double)s; }
    case T_FLOAT64: { unsigned long lo = r_u64(r), hi = r_u64(r);
                      uint64_t v = lo | (hi << 32); double d; memcpy(&d, &v, 8); return d; }
    default:        rfail(r, "неизвестный тип значения в метаданных"); return 0;
    }
}

static size_t type_size(unsigned t)
{
    switch (t) {
    case T_UINT8: case T_INT8: case T_BOOL:   return 1;
    case T_UINT16: case T_INT16:             return 2;
    case T_UINT32: case T_INT32: case T_FLOAT32: return 4;
    case T_UINT64: case T_INT64: case T_FLOAT64: return 8;
    default:                                   return 0;
    }
}

static void meta_push(gguf_file *g, gguf_meta *m, char *key, unsigned type, rd *r)
{
    g->meta[g->n_meta++] = *m;
    memset(m, 0, sizeof *m);
    (void)key;
    (void)type;
    (void)r;
}

static void read_meta(gguf_file *g, rd *r, size_t n)
{
    if (r->bad) return;
    g->meta = xcalloc(n ? n : 1, sizeof *g->meta);

    for (size_t i = 0; i < n && !r->bad; i++) {
        char *key = r_str(r);
        unsigned type = r_u32(r);
        if (r->bad) { free(key); break; }
        if (type > T_FLOAT64) {
            snprintf(r->err, r->errlen, "ключ «%s» имеет тип %u — файл не GGUF", key ? key : "?", type);
            r->bad = 1;
            free(key);
            break;
        }

        gguf_meta m;
        memset(&m, 0, sizeof m);
        m.key = key;
        m.type = type;

        if (type == T_STRING) {
            m.str = r_str(r);
        } else if (type == T_ARRAY) {
            m.arr_type = r_u32(r);
            m.arr_len = r_len(r);
            if (r->bad) { free(key); break; }
            if (m.arr_type == T_STRING) {
                m.arr_str = xcalloc(m.arr_len ? m.arr_len : 1, sizeof *m.arr_str);
                for (size_t k = 0; k < m.arr_len && !r->bad; k++) m.arr_str[k] = r_str(r);
            } else {
                size_t step = type_size(m.arr_type);
                if (!step || m.arr_len * step > r->size) { rfail(r, "плохой массив в метаданных"); break; }
                m.arr_num = xcalloc(m.arr_len ? m.arr_len : 1, sizeof *m.arr_num);
                for (size_t k = 0; k < m.arr_len && !r->bad; k++) m.arr_num[k] = r_num(r, m.arr_type);
            }
        } else {
            m.num = r_num(r, type);
        }
        meta_push(g, &m, key, type, r);
    }
}

int gguf_open(gguf_file *g, const void *file, size_t size, char *err, size_t errlen)
{
    memset(g, 0, sizeof *g);
    g->file = file;
    g->size = size;
    g->alignment = 32;

    rd r = { .p = file, .size = size, .version = 3, .err = err, .errlen = errlen };
    if (size < 24) {
        snprintf(err, errlen, "файл слишком мал для GGUF (%zu байт)", size);
        return -1;
    }
    if (r_u32(&r) != 0x46554747u) {   /* 'GGUF' */
        snprintf(err, errlen, "не GGUF: волшебное число не совпало");
        return -1;
    }
    r.version = r_u32(&r);
    if (r.version < 2 || r.version > 3) {
        snprintf(err, errlen, "версия GGUF %u не поддерживается (нужна 2 или 3)", r.version);
        return -1;
    }
    size_t n_tensors = r_count(&r);
    size_t n_meta = r_count(&r);
    if (r.bad) return -1;
    if (n_tensors == 0 || n_tensors > GGUF_MAX_TENSORS || n_meta > GGUF_MAX_META) {
        snprintf(err, errlen, "GGUF: неправдоподобное число тензоров (%zu) или ключей (%zu)",
                 n_tensors, n_meta);
        return -1;
    }
    g->version = r.version;
    g->n_tensors = n_tensors;
    g->tensors = xcalloc(n_tensors, sizeof *g->tensors);

    read_meta(g, &r, n_meta);
    if (r.bad) goto fail;

    for (size_t i = 0; i < n_tensors && !r.bad; i++) {
        gguf_tensor *t = &g->tensors[i];
        char *name = r_str(&r);
        if (r.bad) { free(name); goto fail; }
        t->name = name;                      /* строки метаданных и тензоров живут в буфере файла */
        t->n_dims = r_u32(&r);
        if (t->n_dims > 4) { rfail(&r, "у тензора больше 4 измерений"); goto fail; }
        size_t total = 1;
        for (unsigned d = 0; d < t->n_dims && !r.bad; d++) {
            t->dims[d] = r_u64(&r);
            if (!t->dims[d]) { rfail(&r, "нулевое измерение тензора"); goto fail; }
            total *= (size_t)t->dims[d];
        }
        t->type = r_u32(&r);
        t->off = (size_t)r_u64(&r);
        if (r.bad) goto fail;
        if (t->type != 0 && t->type != 1) {   /* пока F32 и F16 */
            rfail(&r, "тип тензора поддерживается только F32 и F16");
            goto fail;
        }
        t->nbytes = total * (t->type == 0 ? 4 : 2);
    }
    if (r.bad) goto fail;

    /* выравнивание начала данных задаёт сам файл */
    for (size_t i = 0; i < g->n_meta; i++) {
        const gguf_meta *m = &g->meta[i];
        if (strcmp(m->key, "general.alignment") == 0 && m->num >= 1 && m->num <= 65536)
            g->alignment = (unsigned)m->num;
    }
    /* Смещения тензоров берём из файла, а не укладываем их подряд: в файле
     * между тензорами может быть выравнивание или простои, и последовательная
     * укладка молча сдвинула бы все веса. */
    size_t base = (size_t)(r.p - (const unsigned char *)file);
    base = (base + g->alignment - 1) / g->alignment * g->alignment;

    for (size_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        if (t->off > size || base + t->off + t->nbytes > size) {
            snprintf(err, errlen, "тензор %s выходит за пределы файла", t->name);
            goto fail;
        }
        t->data = (const unsigned char *)file + base + t->off;
    }
    return 0;

fail:
    gguf_close(g);
    return -1;
}

void gguf_close(gguf_file *g)
{
    if (!g) return;
    for (size_t i = 0; i < g->n_meta; i++) {
        free(g->meta[i].key);
        free(g->meta[i].str);
        for (size_t k = 0; k < g->meta[i].arr_len; k++) free(g->meta[i].arr_str[k]);
        free(g->meta[i].arr_str);
        free(g->meta[i].arr_num);
    }
    free(g->meta);
    /* имена тензоров — куски буфера модели, освобождать нечего */
    free(g->tensors);
    memset(g, 0, sizeof *g);
}

const gguf_meta *gguf_find(const gguf_file *g, const char *key)
{
    for (size_t i = 0; i < g->n_meta; i++)
        if (strcmp(g->meta[i].key, key) == 0) return &g->meta[i];
    return NULL;
}

const gguf_tensor *gguf_tensor_find(const gguf_file *g, const char *name)
{
    for (size_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}

const char *gguf_str(const gguf_file *g, const char *key)
{
    const gguf_meta *m = gguf_find(g, key);
    return (m && m->str) ? m->str : NULL;
}

/* Ищет ключ: сначала с префиксом архитектуры, потом без него — в реальных
 * файлах встречаются оба варианта. */
static const gguf_meta *meta_lookup(const gguf_file *g, const char *key, const char *arch)
{
    char full[128];
    const gguf_meta *m = NULL;
    if (arch && *arch) {
        snprintf(full, sizeof full, "%s.%s", arch, key);
        m = gguf_find(g, full);
    }
    if (!m) m = gguf_find(g, key);
    return m;
}

int gguf_u32(const gguf_file *g, unsigned long *out, const char *key, const char *arch)
{
    const gguf_meta *m = meta_lookup(g, key, arch);
    if (!m || m->type == T_STRING || m->type == T_ARRAY) return -1;
    *out = (unsigned long)m->num;
    return 0;
}

int gguf_f32(const gguf_file *g, float *out, const char *key, const char *arch)
{
    const gguf_meta *m = meta_lookup(g, key, arch);
    if (!m || m->type == T_STRING || m->type == T_ARRAY) return -1;
    /* значение читается как double: дробные числа вроде 1e-5 через
     * unsigned long просто обнулились бы */
    *out = (float)m->num;
    return 0;
}

/* F16 → F32: знак, экспонента (5 бит, смещение 15) и мантисса (10 бит). */
static float f16_to_f32(unsigned short h)
{
    unsigned sign = (unsigned)(h >> 15) & 1;
    unsigned exp = (unsigned)(h >> 10) & 0x1F;
    unsigned man = (unsigned)h & 0x3FF;
    uint32_t bits;
    float    out;

    if (exp == 0) {
        if (man == 0) {
            bits = sign << 31;                       /* ноль */
        } else {                                     /* субнормальный F16 */
            int e = -1;
            do { man <<= 1; e++; } while (!(man & 0x400));
            man &= 0x3FF;
            bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = (sign << 31) | 0x7F800000u | (man << 13);   /* бесконечность или NaN */
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    memcpy(&out, &bits, 4);
    return out;
}

int gguf_tensor_to_float(const gguf_tensor *t, float *out, size_t count)
{
    if (!t) return -1;
    if (t->type == 0) {
        const float *src = (const float *)(const void *)t->data;
        for (size_t i = 0; i < count; i++) out[i] = src[i];
        return 0;
    }
    if (t->type == 1) {
        const unsigned short *src = (const unsigned short *)(const void *)t->data;
        for (size_t i = 0; i < count; i++) out[i] = f16_to_f32(src[i]);
        return 0;
    }
    return -1;
}