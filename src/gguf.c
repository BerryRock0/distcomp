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
    const unsigned char *file;  /* начало файла для вычисления позиции */
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

/* Граница читаемости: сколько байт осталось до конца файла. */
static size_t r_left(const rd *r)
{
    return r->size - (size_t)(r->p - r->file);
}

static unsigned r_u8(rd *r)
{
    if (r_left(r) < 1) { rfail(r, "файл оборван"); return 0; }
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

/* Длина строки или массива — ВСЕГДА uint64, независимо от версии файла.
 *
 * Раньше здесь стояло «в v2 — uint64, в v3 — uint32». Это неверно: в v3
 * изменился не размер поля длины, а кое-что другое, и настоящие файлы
 * от llama.cpp (Qwen2.5 и другие) пишут длину ключа восемью байтами.
 * Чтение четырёх байт сдвигает разбор на четыре байта, и дальше вместо
 * ключей читается мусор: «ключ "" имеет тип 1701999988».
 *
 * Счётчики в заголовке файла (число тензоров и ключей) тоже uint64,
 * их читает r_count. */
static size_t r_len(rd *r)
{
    unsigned long v = r_u64(r);
    if (r->bad) return 0;
    size_t remaining = r_left(r);
    if (v > remaining) {
        snprintf(r->err, r->errlen,
                 "строка длиннее файла: заявлено %lu байт, осталось %zu "
                 "(позиция %lu из %zu, версия GGUF %u)",
                 v, remaining,
                 (unsigned long)(r->p - r->file), r->size,
                 r->version);
        r->bad = 1;
        return 0;
    }
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
    if (r_left(r) < n) { rfail(r, "строка обрезана"); free(s); return NULL; }
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

/* Число байт, которое занимает тензор из total элементов.
 *
 * Квантованные веса лежат блоками: в одном блоке не элементов, а сразу
 * несколько штук, плюс к каждому идут свои масштабы и смещения. Поэтому
 * «элементов на байт» здесь нет, и умножать на 4 или 2 нельзя — размер
 * блока задаётся отдельно для каждого типа. Возвращает 0, если тип
 * неизвестен: тогда размер вычислить нельзя, и это не ошибка файла. */
static size_t tensor_nbytes(size_t total, unsigned type)
{
    switch (type) {
    /* обычные типы: байт на элемент */
    case 0:  return total * 4;   /* F32  */
    case 1:  return total * 2;   /* F16  */

    /* K-кванты: 256 элементов в блоке */
    case 10: return total / 256 * 84;    /* Q2_K */
    case 11: return total / 256 * 110;   /* Q3_K */
    case 12: return total / 256 * 144;   /* Q4_K */
    case 13: return total / 256 * 176;   /* Q5_K */
    case 14: return total / 256 * 210;   /* Q6_K */
    case 15: return total / 256 * 292;   /* Q8_K */

    /* старые кванты: 32 элемента в блоке */
    case 2:  return total / 32 * 18;     /* Q4_0 */
    case 3:  return total / 32 * 20;     /* Q4_1 */
    case 6:  return total / 32 * 22;     /* Q5_0 */
    case 7:  return total / 32 * 24;     /* Q5_1 */
    case 8:  return total / 32 * 34;     /* Q8_0 */
    case 9:  return total / 32 * 36;     /* Q8_1 */
    case 16: return total / 32 * 2;      /* IQ4_NL упрощённо — считаем как есть */
    default: return 0;                   /* неизвестный тип: размер неизвестен */
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
            snprintf(r->err, r->errlen,
                     "метаданные: ключ «%s» имеет тип %u (0x%08X) — "
                     "неизвестный тип значения GGUF (позиция %lu из %zu)",
                     key ? key : "?", type, type,
                     (unsigned long)(r->p - r->file), r->size);
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

    rd r = { .p = file, .file = file, .size = size, .version = 3, .err = err, .errlen = errlen };
    if (size < 24) {
        snprintf(err, errlen, "файл слишком мал для GGUF (%zu байт)", size);
        return -1;
    }
    unsigned magic = r_u32(&r);
    if (magic != 0x46554747u) {   /* 'GGUF' */
        snprintf(err, errlen,
                 "не GGUF: волшебное число 0x%08X (ожидалось 0x46554747 'GGUF') — "
                 "файл повреждён или имеет неверный формат",
                 magic);
        return -1;
    }
    r.version = r_u32(&r);
    if (r.version < 2 || r.version > 3) {
        snprintf(err, errlen,
                 "версия GGUF %u не поддерживается (нужна 2 или 3) — "
                 "файл повреждён или имеет неверный формат",
                 r.version);
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
        /* Квантованные веса — норма, а не повод отказываться: у Qwen2.5-Q2_K
         * почти все тензоры имеют тип 10..14. Размер считаем по таблице
         * блоков; для неизвестного типа остаётся 0, и границы файла
         * проверяются по смещению, а не по длине тензора. */
        t->nbytes = tensor_nbytes(total, t->type);
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
        /* arr_str есть только у массивов строк. У числового массива
         * (например tokenizer.ggml.token_type — это 151936 int32) arr_len
         * тоже задан, но arr_str равен NULL, и разыменование arr_str[k]
         * роняло процесс уже после успешного разбора файла. */
        if (g->meta[i].arr_str) {
            for (size_t k = 0; k < g->meta[i].arr_len; k++) free(g->meta[i].arr_str[k]);
            free(g->meta[i].arr_str);
        }
        free(g->meta[i].arr_num);
    }
    free(g->meta);
    /* Имена тензоров тоже копируются в кучу (r_str выделяет память и копирует
     * байты), в отличие от самих весов — те лежат прямо в буфере модели.
     * На 339 тензорах это 7 КБ, которые иначе утекали при каждом открытии. */
    for (size_t i = 0; i < g->n_tensors; i++) free(g->tensors[i].name);
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

/* --------------------------------------------------- деквантизация ------- */
/*
 * K-кванты (типы 10..15) и старые кванты (2..9) раскладываются по блокам,
 * а не построчно: один блок кодирует сразу 256 (K-квант) или 32 (старый
 * квант) весов вместе с общими масштабами. Поэтому «прочитать вес по
 * адресу» здесь не работает — приходится разворачивать целый блок.
 *
 * Раскладка взята из ggml-quants.c (llama.cpp): те же смещения полей,
 * тот же порядок обхода. Непрерывное измерение тензора — ne0 (вход),
 * блоки идут вдоль него.
 */

#define QK_K 256   /* элементов в блоке K-кванта */
#define K_SCALE_SIZE 12

/* little-endian f16 -> float, без обращения к SSE: тот же результат на
 * любой платформе, а float.half на некоторых отсутствует */
static float rd_f16(const unsigned char *p)
{
    unsigned h = (unsigned)p[0] | ((unsigned)p[1] << 8);
    unsigned sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    uint32_t bits;
    float    out;
    if (exp == 0) {
        if (man == 0) {
            bits = sign << 31;
        } else {
            int e = -1;
            do { man <<= 1; e++; } while (!(man & 0x400));
            man &= 0x3FF;
            bits = (sign << 31) | ((uint32_t)(112 - e) << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    memcpy(&out, &bits, 4);
    return out;
}

/* Q2_K: 84 байта на 256 весов. По 4 бита на масштаб в scales[16],
 * по 2 бита на вес в qs[64]. Порядок полей: scales, qs, d, dmin — d лежит
 * на 80-м байте, а не на 64-м (там ещё не конец qs). */
static void deq_q2_K(const unsigned char *x, float *y, int nb)
{
    for (int i = 0; i < nb; i++, x += 84, y += QK_K) {
        const float d = rd_f16(x + 80);
        const float mn = rd_f16(x + 82);
        const unsigned char *sc = x;
        const unsigned char *qs = x + 16;
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            for (int j = 0; j < 4; j++) {
                /* На каждую половину блока приходится два байта qs:
                 * j=0,1 читают первый, j=2,3 — второй. Раньше брался один
                 * байт на все четыре группы, и веса внутри блока вставали
                 * на места попарно — снаружи это не видно, но Q/K и FFN
                 * после этого считались по переставленным весам. */
                unsigned char q = qs[n / 128 + (j >= 2 ? 1 : 0)];
                int shift = j * 4;
                unsigned char s1 = sc[is++];
                float d1 = d * (s1 & 0xF), m1 = mn * (s1 >> 4);
                for (int l = 0; l < 16; l++) y[n + j * 32 + l] = d1 * ((q >> shift) & 3) - m1;
                unsigned char s2 = sc[is++];
                float d2 = d * (s2 & 0xF), m2 = mn * (s2 >> 4);
                for (int l = 0; l < 16; l++)
                    y[n + j * 32 + 16 + l] = d2 * ((q >> (shift + 2)) & 3) - m2;
            }
        }
    }
}

/* Q3_K: 110 байт. 3 бита на вес: младшие два в qs[64], старший бит в
 * hmask[32]. Масштабы 6 бит, упакованы по 4 штуки в байт (12 байт). */
static void deq_q3_K(const unsigned char *x, float *y, int nb)
{
    static const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    for (int i = 0; i < nb; i++, x += 110, y += QK_K) {
        const float d = rd_f16(x + 108);
        const unsigned char *hmask = x;
        const unsigned char *qs = x + 32;
        const unsigned char *scraw = x + 96;

        uint32_t aux[4];
        memcpy(aux, scraw, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        /* Масштабов восемь, а не четыре: в каждой половине блока по четыре
 * группы по 32 веса. Обращение к dl[2+j] при массиве из четырёх элементов
 * уезжало за пределы и давало значения порядка миллиона. */
        int8_t ls[8];
        float  dl[8];
        for (int j = 0; j < 4; j++) {
            int v = (int)(aux[j] & 0xF) | (int)(((j < 2 ? aux[j] : aux[j - 2]) >> 4) & 0x3) << 4;
            ls[j] = (int8_t)(v - 32);
            dl[j] = d * (float)ls[j];
            int v2 = (int)(aux[j + 4] & 0xF) |
                     (int)(((j < 2 ? aux[j + 4] : aux[j + 2]) >> 4) & 0x3) << 4;
            ls[j + 4] = (int8_t)(v2 - 32);
            dl[j + 4] = d * (float)ls[j + 4];
        }

        unsigned m = 1;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl0 = dl[j], dl1 = dl[2 + j];
                for (int l = 0; l < 16; l++) {
                    int q0 = (qs[n / 128 * 32 + l] >> shift) & 3;
                    int q1 = (qs[n / 128 * 32 + l + 16] >> shift) & 3;
                    y[n + j * 32 + l] = dl0 * (q0 - ((hmask[n / 128 * 32 + l] & m) ? 0 : 4));
                    y[n + j * 32 + 16 + l] = dl1 * (q1 - ((hmask[n / 128 * 32 + l + 16] & m) ? 0 : 4));
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
}

/* Q4_K: 144 байта. 4 бита на вес в qs[128], 12 байт масштабов/сдвигов
 * (6 бит + 6 бит на группу из 32 весов). */
static void get_scale_min_k4(int j, const unsigned char *q, unsigned char *sc, unsigned char *m)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (unsigned char)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m  = (unsigned char)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static void deq_q4_K(const unsigned char *x, float *y, int nb)
{
    for (int i = 0; i < nb; i++, x += 144, y += QK_K) {
        const float d = rd_f16(x);
        const float mn = rd_f16(x + 2);
        const unsigned char *sc = x + 4;
        const unsigned char *qs = x + 16;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            unsigned char s, m;
            get_scale_min_k4(is + 0, sc, &s, &m);
            float d1 = d * s, m1 = mn * m;
            get_scale_min_k4(is + 1, sc, &s, &m);
            float d2 = d * s, m2 = mn * m;
            for (int l = 0; l < 32; l++) y[j + l]     = d1 * (qs[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) y[j + 32 + l] = d2 * (qs[l] >> 4)  - m2;
            qs += 32;
            is += 2;
        }
    }
}

/* Q5_K: 176 байт. Как Q4_K, плюс 32 байта старших битов в qh[32]. */
static void deq_q5_K(const unsigned char *x, float *y, int nb)
{
    for (int i = 0; i < nb; i++, x += 176, y += QK_K) {
        const float d = rd_f16(x);
        const float mn = rd_f16(x + 2);
        const unsigned char *sc = x + 4;
        const unsigned char *qh = x + 16;
        const unsigned char *qs = x + 48;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            unsigned char s, m;
            get_scale_min_k4(is + 0, sc, &s, &m);
            float d1 = d * s, m1 = mn * m;
            get_scale_min_k4(is + 1, sc, &s, &m);
            float d2 = d * s, m2 = mn * m;
            for (int l = 0; l < 32; l++)
                y[j + l] = d1 * ((int)((qs[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 16) - m1;
            for (int l = 0; l < 32; l++)
                y[j + 32 + l] = d2 * ((int)((qs[l] >> 4) | (((qh[l] >> 2) & 3) << 4)) - 16) - m2;
            qs += 32;
            is += 2;
        }
    }
}

/* Q6_K: 210 байт. 4 младших бита в ql[128], 2 старших в qh[64],
 * 16 масштабов int8 и общий f16. */
static void deq_q6_K(const unsigned char *x, float *y, int nb)
{
    for (int i = 0; i < nb; i++, x += 210, y += QK_K) {
        const float d = rd_f16(x + 208);
        const unsigned char *ql = x;
        const unsigned char *qh = x + 128;
        const signed char *sc = (const signed char *)(const void *)(x + 192);
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[n / 128 * 64 + l] & 0xF) |
                                     (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[n / 128 * 64 + l + 32] & 0xF) |
                                     (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[n / 128 * 64 + l] >> 4) |
                                     (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[n / 128 * 64 + l + 32] >> 4) |
                                     (((qh[l] >> 6) & 3) << 4)) - 32;
                y[n + l]      = d * sc[is + 0] * q1;
                y[n + l + 32] = d * sc[is + 2] * q2;
                y[n + 64 + l] = d * sc[is + 4] * q3;
                y[n + 96 + l] = d * sc[is + 6] * q4;
            }
        }
    }
}

/* Q8_K: 292 байта: 256 int8 + 32 байта масштабов (по f16 на каждые 32). */
static void deq_q8_K(const unsigned char *x, float *y, int nb)
{
    for (int i = 0; i < nb; i++, x += 292, y += QK_K) {
        const signed char *q = (const signed char *)(const void *)x;
        for (int l = 0; l < 32; l++) {
            float d = rd_f16(x + 256 + l * 2);
            for (int m = 0; m < 8; m++) y[l * 8 + m] = d * (float)q[l * 8 + m];
        }
    }
}

/* Старые кванты: 32 веса в блоке. */
static void deq_q4_0(const unsigned char *x, float *y, int nb)
{
    for (int i = 0; i < nb; i++, x += 18, y += 32) {
        float d = rd_f16(x);
        for (int l = 0; l < 16; l++) {
            y[l]     = ((int)(x[2 + l] & 0xF) - 8) * d;
            y[l + 16] = ((int)(x[2 + l] >> 4) - 8) * d;
        }
    }
}

static void deq_q4_1(const unsigned char *x, float *y, int nb)
{
    for (int i = 0; i < nb; i++, x += 20, y += 32) {
        float d = rd_f16(x), m = rd_f16(x + 2);
        for (int l = 0; l < 16; l++) {
            y[l]     = (int)(x[4 + l] & 0xF) * d + m;
            y[l + 16] = (int)(x[4 + l] >> 4) * d + m;
        }
    }
}

static void deq_q5_0(const unsigned char *x, float *y, int nb)
{
    for (int i = 0; i < nb; i++, x += 22, y += 32) {
        float d = rd_f16(x);
        unsigned qh = (unsigned)x[2] | ((unsigned)x[3] << 8) |
                      ((unsigned)x[4] << 16) | ((unsigned)x[5] << 24);
        for (int l = 0; l < 16; l++) {
            int hl = (int)((qh >> l) & 1) << 4;
            int h2 = (int)((qh >> (l + 16)) & 1) << 4;
            y[l]     = (((int)(x[6 + l] & 0xF) | hl) - 16) * d;
            y[l + 16] = (((int)(x[6 + l] >> 4) | h2) - 16) * d;
        }
    }
}

static void deq_q5_1(const unsigned char *x, float *y, int nb)
{
    for (int i = 0; i < nb; i++, x += 24, y += 32) {
        float d = rd_f16(x), m = rd_f16(x + 2);
        unsigned qh = (unsigned)x[4] | ((unsigned)x[5] << 8) |
                      ((unsigned)x[6] << 16) | ((unsigned)x[7] << 24);
        for (int l = 0; l < 16; l++) {
            int hl = (int)((qh >> l) & 1) << 4;
            int h2 = (int)((qh >> (l + 16)) & 1) << 4;
            y[l]     = ((int)(x[8 + l] & 0xF) | hl) * d + m;
            y[l + 16] = ((int)(x[8 + l] >> 4) | h2) * d + m;
        }
    }
}

static void deq_q8_0(const unsigned char *x, float *y, int nb)
{
    for (int i = 0; i < nb; i++, x += 34, y += 32) {
        float d = rd_f16(x);
        for (int l = 0; l < 32; l++) y[l] = d * (float)(signed char)x[2 + l];
    }
}

/* Сколько всего элементов в тензоре. */
static size_t tensor_elems(const gguf_tensor *t)
{
    size_t n = 1;
    for (unsigned i = 0; i < t->n_dims && i < 4; i++) n *= (size_t)t->dims[i];
    return n;
}

/* Разворачивает элементы [off, off+n) тензора в float.
 * Для F32/F16 это копирование; для квантов — по блокам, поэтому off обязан
 * попадать на границу блока (для K-квантов — кратным 256). */
int gguf_dequant(const gguf_tensor *t, float *dst, size_t off, size_t n)
{
    if (!t || !dst) return -1;
    size_t total = tensor_elems(t);
    if (off > total || n > total - off) return -1;
    if (n == 0) return 0;

    const unsigned char *raw = t->data;

    if (t->type == 0) {
        memcpy(dst, raw + off * 4, n * 4);
        return 0;
    }
    if (t->type == 1) {
        const unsigned short *h = (const unsigned short *)(const void *)raw + off;
        for (size_t i = 0; i < n; i++) dst[i] = rd_f16((const unsigned char *)(h + i));
        return 0;
    }

    /* кванты: блок по QK_K (или 32), off обязан быть на границе блока */
    size_t blk, first;
    switch (t->type) {
    case 10: case 11: case 12: case 13: case 14: case 15:
        blk = QK_K;
        if (off % blk) return -1;
        first = off / blk;
        break;
    case 2: case 3: case 6: case 7: case 8:
        blk = 32;
        if (off % blk) return -1;
        first = off / blk;
        break;
    default:
        return -1;   /* IQ-типы и прочее не поддержаны */
    }

    size_t nblk = (n + blk - 1) / blk;
    if (off / blk + nblk > total / blk) return -1;

    switch (t->type) {
    case 10: deq_q2_K(raw + first *  84, dst, (int)nblk); return 0;
    case 11: deq_q3_K(raw + first * 110, dst, (int)nblk); return 0;
    case 12: deq_q4_K(raw + first * 144, dst, (int)nblk); return 0;
    case 13: deq_q5_K(raw + first * 176, dst, (int)nblk); return 0;
    case 14: deq_q6_K(raw + first * 210, dst, (int)nblk); return 0;
    case 15: deq_q8_K(raw + first * 292, dst, (int)nblk); return 0;
    case  2: deq_q4_0(raw + first *  18, dst, (int)nblk); return 0;
    case  3: deq_q4_1(raw + first *  20, dst, (int)nblk); return 0;
    case  6: deq_q5_0(raw + first *  22, dst, (int)nblk); return 0;
    case  7: deq_q5_1(raw + first *  24, dst, (int)nblk); return 0;
    case  8: deq_q8_0(raw + first *  34, dst, (int)nblk); return 0;
    default: return -1;
    }
}

int gguf_quant_type(const gguf_file *g)
{
    for (size_t i = 0; i < g->n_tensors; i++)
        if (g->tensors[i].type != 0 && g->tensors[i].type != 1) return (int)g->tensors[i].type;
    return -1;
}

int gguf_tensor_to_float(const gguf_tensor *t, float *out, size_t count)
{
    return gguf_dequant(t, out, 0, count);
}