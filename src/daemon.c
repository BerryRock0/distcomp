#include "conf.h"
#include "expr.h"
#include "engine.h"
#include "udp.h"
#include "util.h"

#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/*
 * dcd — минимальный узел распределённых вычислений.
 *
 * Один цикл, ни одного потока и ни одного malloc: все буферы лежат в BSS.
 * Поэтому один и тот же код собирается и на сервер, и на микроконтроллер —
 * SAMD51, RP2040, ESP32, STM32: меняется только размер буфера сборки
 * (DC_UDP_MAX_DGRAM). Потоки убраны не ради красоты: задача узла —
 * посчитать арифметику или дописать строку, это микросекунды, а pthreads
 * есть далеко не везде.
 *
 * Узел обслуживает одну или несколько ролей сразу (на устройстве с 40 КБ
 * ОЗУ три процесса не запустить, а один с тремя ролями помещается):
 *   compute   — считает арифметическое выражение из payload
 *   verify    — сверяет результаты с payload (числом или "all")
 *   memorize  — дописывает JSONL-запись в state/memory.log
 *
 * Протокол — UDP: 14-байтовый заголовок + JSON (см. udp.h). Форматы
 * сообщений совпадают с прошлыми версиями, поэтому планировщик работает
 * и со старыми узлами, и с новыми.
 */

/* Сколько байт тела узел принимает по умолчанию — столько, сколько влезает
 * в одну дейтаграмму. Сборка lite задаёт DC_MAX_BODY меньше: буфер на 32 КБ
 * ОЗУ дорог, а принимать всё равно больше негде. */
#ifndef DC_MAX_BODY
#define DC_MAX_BODY (DC_UDP_MAX_DGRAM - DC_UDP_HEADER_LEN)
#endif
#define DC_MAX_CTX  16

#define STR2(x) #x
#define STR(x)  STR2(x)

#ifndef DC_ARCH
#if defined(__aarch64__)
#define DC_ARCH "aarch64"
#elif defined(__arm__)
#define DC_ARCH "arm"
#elif defined(__x86_64__)
#define DC_ARCH "x86_64"
#elif defined(__riscv)
#define DC_ARCH "riscv"
#elif defined(__AVR__)
#define DC_ARCH "avr"
#else
#define DC_ARCH "posix"
#endif
#endif

#ifndef DC_HELLO_SECS
#define DC_HELLO_SECS 10
#endif

/* Сколько байт весов узел готов держать в ОЗУ под части модели. Веса
 * лежат в статическом пуле: узел по построению работает без malloc,
 * а веса должны жить в памяти постоянно, а не выделяться под каждый
 * запрос. Сборка lite задаёт своё значение. */
#ifndef DC_SHARD_MAX_BYTES
#define DC_SHARD_MAX_BYTES (1u << 20)
#endif
#define DC_SHARD_LAYERS 16   /* сколько слоёв узел держит одновременно */

/* ---------------------------------------------------------------- узел -- */

typedef struct {
    int  port;
    int  max_body;
    int  serve[4];
    int  nroles;
    char tag[32];
    char state_dir[256];
    long served;
    long free_mem;          /* сколько пула отдаём под модель */
} node;

static node N;   /* единственный экземпляр — в BSS, без malloc */

static const char *role_name(int idx)
{
    static const char *names[] = { ROLE_COMPUTE, ROLE_VERIFY, ROLE_MEMORIZE, ROLE_SHARD };
    return idx >= 0 && idx < 4 ? names[idx] : "?";
}

static int role_index(const char *name)
{
    if (str_ieq(name, ROLE_COMPUTE)) return 0;
    if (str_ieq(name, ROLE_VERIFY)) return 1;
    if (str_ieq(name, ROLE_MEMORIZE)) return 2;
    if (str_ieq(name, ROLE_SHARD)) return 3;
    return -1;
}

/* --------------------------------------------- строковые помощники -- */
/* Вместо snprintf — три крошечные функции: узлу нужны только склейка
 * строк и дописывание числа. */

static char *scpy(char *dst, size_t dstsz, const char *src)
{
    size_t o = 0;
    if (!dstsz) return dst;
    for (; src[o] && o + 1 < dstsz; o++) dst[o] = src[o];
    dst[o] = '\0';
    return dst;
}

static char *scat(char *dst, size_t dstsz, const char *src)
{
    size_t o = strlen(dst);
    if (o + 1 >= dstsz) return dst;
    for (; *src && o + 1 < dstsz; o++) dst[o] = *src++;
    dst[o] = '\0';
    return dst;
}

static char *scat_u(char *dst, size_t dstsz, unsigned long v)
{
    char tmp[24];
    int  i = 0;
    if (!v) return scat(dst, dstsz, "0");
    while (v) { tmp[i++] = (char)('0' + v % 10); v /= 10; }
    size_t o = strlen(dst);
    while (i && o + 1 < dstsz) dst[o++] = tmp[--i];
    dst[o] = '\0';
    return dst;
}

/* Читает неотрицательное целое и сдвигает курсор на конец числа.
 * Свой разбор вместо strtol: на микроконтроллере это килобайты. */
static int sint(const char *p, const char **end)
{
    int v = 0;
    int any = 0;
    while (*p >= '0' && *p <= '9') {
        v = v * 10 + (*p - '0');
        p++;
        any = 1;
        if (v > 100000000) v = 100000000;   /* защита от переполнения */
    }
    if (end) *end = p;
    return any ? v : -1;
}

/* -------------------------------------------------- минимальный вывод -- */
/* Свой printf: %s %d %u %ld %zu %%. Без printf-семейства libc — на маленьких
 * платформах оно тянет за собой килобайты кода. На MCU достаточно заменить
 * одну функцию dput_all на отправку в UART. */

static void dput_all(const char *s, size_t n)
{
    if (n && write(STDOUT_FILENO, s, n) < 0) { /* UART на MCU */ }
}

static void dprint(const char *fmt, ...)
{
    char    buf[256];
    size_t  o = 0;
    va_list ap;
    va_start(ap, fmt);
    for (const char *p = fmt; *p && o + 24 < sizeof buf; p++) {
        if (*p != '%') { buf[o++] = *p; continue; }
        p++;
        char num[24];
        if (*p == 'l') {              /* %ld */
            p++;
            long v = va_arg(ap, long);
            int neg = v < 0;
            unsigned long u = neg ? -(unsigned long)v : (unsigned long)v;
            int i = 0;
            do { num[i++] = (char)('0' + u % 10); u /= 10; } while (u);
            if (neg) num[i++] = '-';
            while (i && o + 1 < sizeof buf) buf[o++] = num[--i];
        } else if (*p == 'z') {       /* %zu */
            p++;
            unsigned long u = va_arg(ap, size_t);
            int i = 0;
            do { num[i++] = (char)('0' + u % 10); u /= 10; } while (u);
            while (i && o + 1 < sizeof buf) buf[o++] = num[--i];
        } else if (*p == 'd' || *p == 'i' || *p == 'u') {   /* %d %i %u */
            unsigned long u = va_arg(ap, unsigned int);
            int i = 0;
            do { num[i++] = (char)('0' + u % 10); u /= 10; } while (u);
            while (i && o + 1 < sizeof buf) buf[o++] = num[--i];
        } else if (*p == 's') {       /* %s */
            const char *s = va_arg(ap, const char *);
            while (*s && o + 1 < sizeof buf) buf[o++] = *s++;
        } else if (*p == '%') {
            buf[o++] = '%';
        } else {
            buf[o++] = '%';
            if (o + 1 < sizeof buf) buf[o++] = *p;
        }
    }
    va_end(ap);
    buf[o] = '\0';
    dput_all(buf, o);
}

/* ------------------------------------------------------- буфер ответа -- */
/* Буфер фиксированного размера: переполнение — обрезка, а не падение. */

typedef struct {
    char  *p;
    size_t len, cap;
} buf_t;

static void binit(buf_t *b, char *p, size_t cap)
{
    b->p = p;
    b->len = 0;
    b->cap = cap;
    if (cap) p[0] = '\0';
}

static void badd(buf_t *b, const char *s)
{
    while (*s && b->len + 1 < b->cap) b->p[b->len++] = *s++;
    b->p[b->len] = '\0';
}

/* Строка как JSON-строка: в кавычках, экранирует " и \, остальное как есть. */
static void badd_json(buf_t *b, const char *s)
{
    if (!s) return;
    badd(b, "\"");
    for (; *s && b->len + 1 < b->cap; s++) {
        if (*s == '"' || *s == '\\') {
            if (b->len + 2 >= b->cap) break;
            b->p[b->len++] = '\\';
        }
        b->p[b->len++] = *s;
    }
    b->p[b->len] = '\0';
    badd(b, "\"");
}

static void badd_u(buf_t *b, unsigned long v)
{
    char tmp[24];
    int  i = 0;
    if (!v) { badd(b, "0"); return; }
    while (v) { tmp[i++] = (char)('0' + v % 10); v /= 10; }
    while (i && b->len + 1 < b->cap) b->p[b->len++] = tmp[--i];
    b->p[b->len] = '\0';
}

/* Ответ и строка журнала собираются здесь: оба заведомо меньше приёма.
 * Всё в BSS, как и обещано в шапке файла: стек на микроконтроллере часто
 * меньше, чем одно тело запроса. */
static char rx_buf[DC_MAX_BODY + 1];
static char reply_buf[DC_MAX_BODY + 512];

/* ------------------------------------------------------- JSON-сканер ---- */
/* Полноценный парсер (json.c) узлу не нужен: запросы к нему имеют
 * фиксированную форму (build_request_body в sched.c). Этот сканер не
 * аллоцирует ничего и не умеет глубже одного уровня — только поля
 * role/title/payload верхнего объекта и from/output элементов context.
 *
 * Запрос:
 * {"role":"compute","title":"...","payload":"...","attempt":1,
 *  "context":[{"from":"s1","role":"compute","output":"3750"}]} */

typedef struct { const char *p; } js;

static void js_ws(js *j)
{
    while (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r') j->p++;
}

/* Строка, в позиции j стоит открывающая кавычка. out==NULL — только проход.
 * Простые escape расподобляет, \uXXXX отдаёт сырым текстом (планировщик
 * разберёт сам). Обрезка по длине — молча: поля короткие. */
static int js_str(js *j, char *out, size_t outsz)
{
    if (*j->p != '"') return 0;
    j->p++;
    size_t o = 0;
    while (*j->p && *j->p != '"') {
        unsigned char ch = (unsigned char)*j->p;
        if (ch == '\\' && j->p[1]) {
            j->p++;
            char e = *j->p++;
            switch (e) {
            case 'n': ch = '\n'; break;
            case 't': ch = '\t'; break;
            case 'r': ch = '\r'; break;
            case 'b': ch = '\b'; break;
            case 'f': ch = '\f'; break;
            case '"':  ch = '"';  break;
            case '\\': ch = '\\'; break;
            case '/':  ch = '/';  break;
            case 'u':
                if (out && o + 6 < outsz) { out[o++] = '\\'; out[o++] = 'u'; }
                for (int i = 0; i < 4 && *j->p; i++) {
                    if (out && o + 1 < outsz) out[o++] = *j->p;
                    j->p++;
                }
                continue;
            default: ch = (unsigned char)e; break;
            }
        } else {
            j->p++;
        }
        if (out && o + 1 < outsz) out[o++] = (char)ch;
    }
    if (*j->p == '"') j->p++;
    if (out) out[o] = '\0';
    return 1;
}

/* Пропускает любое значение: строку, число, литерал или контейнер. */
static void js_skip_value(js *j)
{
    js_ws(j);
    char c = *j->p;
    if (c == '"') { js_str(j, NULL, 0); return; }
    if (c == '{' || c == '[') {
        int depth = 0;
        for (;;) {
            char d = *j->p;
            if (!d) return;
            if (d == '"') { js_str(j, NULL, 0); continue; }
            if (d == '{' || d == '[') depth++;
            else if (d == '}' || d == ']') {
                depth--;
                j->p++;
                if (depth <= 0) return;
                continue;
            }
            j->p++;
        }
    }
    /* число или литерал: до разделителя */
    while (*j->p && *j->p != ',' && *j->p != '}' && *j->p != ']' &&
           *j->p != ' ' && *j->p != '\t' && *j->p != '\n' && *j->p != '\r')
        j->p++;
}

/* Читает строковое поле key объекта, в позиции j — его начало.
 *
 * Поля перебираются все до конца, а не только первое: искомое поле может
 * стоять где угодно («{"from":"s1","role":"compute","output":"3750"}» —
 * output здесь второе). */
static int js_obj_string(js *j, const char *key, char *out, size_t outsz)
{
    js_ws(j);
    if (*j->p != '{') return 0;
    j->p++;
    char kbuf[48];
    for (;;) {
        js_ws(j);
        if (*j->p == '}' || !*j->p) return 0;   /* '}' или конец — поля кончились */
        if (*j->p != '"') return 0;             /* мусор в объекте */
        if (!js_str(j, kbuf, sizeof kbuf)) return 0;
        js_ws(j);
        if (*j->p != ':') return 0;
        j->p++;
        js_ws(j);
        /* значение искомого поля обязано быть строкой: повторного поля с тем
         * же именем в объекте не бывает, искать дальше смысла нет */
        if (strcmp(kbuf, key) == 0) return *j->p == '"' ? js_str(j, out, outsz) : 0;
        js_skip_value(j);
        js_ws(j);
        if (*j->p == ',') { j->p++; continue; }
        return 0;
    }
}

/* То же, но каждый раз с начала объекта и без побочных эффектов на курсоре:
 * один объект читается несколько раз подряд (from и output), и второй проход
 * не должен зависеть от того, где кончился первый. */
static int js_obj_get(const char *obj, const char *key, char *out, size_t outsz)
{
    js j = { obj };
    return js_obj_string(&j, key, out, outsz);
}

/* Вызывает cb(idx, from, output) для каждого объекта массива-поля key. */
typedef void (*js_ctx_cb)(int idx, const char *from, const char *output, void *ud);

static int js_obj_array(js *j, const char *key, js_ctx_cb cb, void *ud)
{
    js_ws(j);
    if (*j->p != '{') return 0;
    j->p++;
    char kbuf[48];
    for (;;) {
        js_ws(j);
        if (*j->p != '"') return 0;
        if (!js_str(j, kbuf, sizeof kbuf)) return 0;
        js_ws(j);
        if (*j->p != ':') return 0;
        j->p++;
        if (strcmp(kbuf, key) == 0) {
            js_ws(j);
            if (*j->p != '[') return 0;
            j->p++;
            int idx = 0;
            for (;;) {
                js_ws(j);
                if (*j->p == ']' || !*j->p) return 1;
                if (*j->p == '{') {
                    /* объект контекста читается с начала целиком: поля from и
                     * output стоят рядом, но искать их нужно независимо */
                    const char *el = j->p;
                    char from[32] = "", output[128] = "";
                    js_obj_get(el, "from", from, sizeof from);
                    js_obj_get(el, "output", output, sizeof output);
                    if (cb) cb(idx, from, output, ud);
                    idx++;
                }
                js_skip_value(j);
                js_ws(j);
                if (*j->p == ',') { j->p++; continue; }
                return *j->p == ']';
            }
        }
        js_skip_value(j);
        js_ws(j);
        if (*j->p == ',') { j->p++; continue; }
        return 0;
    }
}

/* Разобранный запрос. payload — во весь размер дейтаграммы: выражения
 * бывают длинными (лимит lite-узла — как раз про это). */
typedef struct {
    char role[16];
    char title[256];
    char payload[DC_MAX_BODY + 1];
    int  nctx;
    char ctx_from[DC_MAX_CTX][32];
    char ctx_output[DC_MAX_CTX][128];
} req_t;

static void req_ctx_cb(int idx, const char *from, const char *output, void *ud)
{
    req_t *r = ud;
    if (idx >= DC_MAX_CTX) return;
    if (from && *from) scpy(r->ctx_from[idx], sizeof r->ctx_from[idx], from);
    if (output && *output) scpy(r->ctx_output[idx], sizeof r->ctx_output[idx], output);
    r->nctx = idx + 1;
}

/* Разбирает тело запроса. 1 — роль распознана. */
static int req_parse(const char *json, req_t *r)
{
    memset(r, 0, sizeof *r);
    js_obj_get(json, "role", r->role, sizeof r->role);
    js_obj_get(json, "title", r->title, sizeof r->title);
    js_obj_get(json, "payload", r->payload, sizeof r->payload);

    js jc = { json };
    js_obj_array(&jc, "context", req_ctx_cb, r);
    return r->role[0] != '\0';
}

/* -------------------------------------------------------- роли ---- */

static int role_allowed(int idx)
{
    for (int i = 0; i < N.nroles; i++)
        if (N.serve[i] == idx) return 1;
    return 0;
}

static void role_compute(const req_t *r, char *out, size_t outsz, char *detail,
                         size_t detailsz, int *ok)
{
    const char *payload = r->payload[0] ? r->payload : (r->title[0] ? r->title : "");
    char   err[128] = {0};
    double v = 0;
    if (expr_eval(payload, &v, err, sizeof err) == 0) {
        char num[48];
        expr_format(v, num, sizeof num);
        scpy(out, outsz, num);
        scpy(detail, detailsz, payload);
        scat(detail, detailsz, " = ");
        scat(detail, detailsz, num);
        *ok = 1;
        return;
    }
    /* Не арифметика — честный отказ, а не хеш: число-результат, не имеющее
     * отношения к задаче, спрятал бы ошибку до следующего шага verify. */
    scpy(out, outsz, "ОШИБКА");
    scpy(detail, detailsz, err);
    *ok = 0;
}

/* verify: все результаты непустые; если payload — число, один из них
 * должен с ним сойтись. payload "all" — только непустота. */
static void role_verify(const req_t *r, char *out, size_t outsz, char *detail,
                        size_t detailsz, int *ok)
{
    if (r->nctx == 0) {
        scpy(out, outsz, "ПРОВАЛ");
        scpy(detail, detailsz, "нет входных данных");
        *ok = 0;
        return;
    }
    int nonempty = 0;
    for (int i = 0; i < r->nctx; i++)
        if (r->ctx_output[i][0]) nonempty++;
    if (nonempty != r->nctx) {
        scpy(out, outsz, "ПРОВАЛ");
        scpy(detail, detailsz, "пустых результатов: ");
        scat_u(detail, detailsz, (unsigned long)(r->nctx - nonempty));
        *ok = 0;
        return;
    }

    double want = 0;
    char   eerr[128] = {0};
    int    numeric = !str_ieq(r->payload, "all") &&
                     expr_eval(r->payload, &want, eerr, sizeof eerr) == 0;
    if (!numeric) {
        scpy(out, outsz, "OK");
        scpy(detail, detailsz, "непустых результатов: ");
        scat_u(detail, detailsz, (unsigned long)r->nctx);
        *ok = 1;
        return;
    }
    for (int i = 0; i < r->nctx; i++) {
        /* ctx_output всегда строка (возможно пустая) — strtod(NULL) не будет */
        double got = strtod(r->ctx_output[i], NULL);
        if (fabs(got - want) < 1e-9) {
            scpy(out, outsz, "OK");
            scpy(detail, detailsz, "совпало с ");
            scat(detail, detailsz, r->ctx_from[i][0] ? r->ctx_from[i] : "?");
            *ok = 1;
            return;
        }
    }
    char wantstr[48];
    expr_format(want, wantstr, sizeof wantstr);
    scpy(out, outsz, "ПРОВАЛ");
    scpy(detail, detailsz, "ожидалось ");
    scat(detail, detailsz, wantstr);
    *ok = 0;
}

/* memorize: дописывает JSONL-строку. Счётчик строк держим в памяти:
 * пересчитывать файл при каждой записи дорого, а точность нужна только
 * для журнала. */
static char mpath[512];
static long mem_count = -1;   /* -1 = ещё не посчитано */

static void mem_count_init(void)
{
    mem_count = 0;
    FILE *f = fopen(mpath, "r");
    if (!f) return;
    char lbuf[256];
    while (fgets(lbuf, sizeof lbuf, f)) mem_count++;
    fclose(f);
}

static void role_memorize(const req_t *r, char *out, size_t outsz, char *detail,
                          size_t detailsz, int *ok)
{
    buf_t b;
    binit(&b, reply_buf, sizeof reply_buf);
    char ts[32];
    time_t    now = time(NULL);
    struct tm tmv;
    localtime_r(&now, &tmv);
    strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", &tmv);

    badd(&b, "{\"ts\":\"");
    badd(&b, ts);
    badd(&b, "\",\"role\":\"memorize\",\"node\":");
    badd_json(&b, N.tag);
    badd(&b, ",\"title\":");
    badd_json(&b, r->title[0] ? r->title : "");
    badd(&b, ",\"payload\":");
    badd_json(&b, r->payload[0] ? r->payload : "");
    badd(&b, ",\"context\":[");
    for (int i = 0; i < r->nctx; i++) {
        if (i) badd(&b, ",");
        badd(&b, "{\"from\":");
        badd_json(&b, r->ctx_from[i]);
        badd(&b, ",\"output\":");
        badd_json(&b, r->ctx_output[i]);
        badd(&b, "}");
    }
    badd(&b, "]}\n");

    FILE *f = fopen(mpath, "a");
    if (!f) {
        scpy(out, outsz, "ОШИБКА");
        scpy(detail, detailsz, "нет записи: ");
        scat(detail, detailsz, mpath);
        *ok = 0;
        return;
    }
    int written = fwrite(b.p, 1, b.len, f) == b.len;
    fclose(f);
    if (!written) {
        scpy(out, outsz, "ОШИБКА");
        scpy(detail, detailsz, "нет записи: ");
        scat(detail, detailsz, mpath);
        *ok = 0;
        return;
    }
    mem_count++;
    scpy(out, outsz, "записано #");
    scat_u(out, outsz, (unsigned long)mem_count);
    scpy(detail, detailsz, "всего записей: ");
    scat_u(detail, detailsz, (unsigned long)mem_count);
    *ok = 1;
}

/* ----------------------------------------------------------------- llm --- */
/*
 * Часть LLM на этом устройстве.
 *
 * Узел не знает про GGUF и не парсит его: планировщик присылает тензоры
 * по одному — имя, размеры, тип, байты кусками по max_body, — а узел просто
 * раскладывает их в статический пул. Команда shape объявляет, какую стадию
 * модельного конвейера это устройство обслуживает (вход, слои или выход), и
 * узел сам собирает рабочий контекст из того, что лежит в пуле.
 *
 * Ни одного malloc: пул, арена под активации и результат лежат в BSS. Поэтому
 * тот же код собирается и на сервере, и на микроконтроллере.
 *
 * Протокол узла:
 *   info                                  что держим и сколько свободно
 *   free                                  освободить пул
 *   wt|имя|ndim|d0,d1,..|тип|off|total|b64   кусок тензора в пул
 *   shape|stage|first|last|...           собрать контекст
 *   reset                                 сбросить KV-кэш
 *   gen|токен                             один шаг, результат кэшируется
 *   part|i|n                              i-й кусок результата по n байт
 */

#define DC_SHARD_TENSORS 48
#define DC_SHARD_NAME    48

#ifndef DC_SHARD_RESB
#define DC_SHARD_RESB 16384          /* максимум float в результате шага */
#endif
#ifndef DC_SHARD_ARENA
#define DC_SHARD_ARENA  (256 * 1024)  /* float под активации и KV-кэш */
#endif

typedef struct {
    char            name[DC_SHARD_NAME];
    size_t         off;              /* смещение в пуле */
    size_t         nbytes;
    unsigned       type;             /* 0 = F32, 1 = F16 */
    unsigned       n_dims;
    unsigned long  dims[4];
} shard_tensor;

static unsigned char shard_pool[DC_SHARD_MAX_BYTES];
static shard_tensor  shard_t[DC_SHARD_TENSORS];
static int           shard_nt = 0;
static size_t        shard_used = 0;

/* собранный из пула «GGUF на минималках»: движок не должен знать, что файла нет */
static gguf_file    shard_g;
static gguf_tensor  shard_gt[DC_SHARD_TENSORS];
static gguf_meta    shard_gm[16];
static int          shard_gm_n;

static llm_ctx       shard_ctx;
static llm_layer     shard_layers[DC_SHARD_LAYERS];
static float         shard_arena[DC_SHARD_ARENA];
static int           shard_ready;       /* контекст собран */
static int           shard_res_n;       /* float в последнем результате */
static int           shard_res_tok;     /* для стадии выхода: выбранный токен */
static float         shard_res[DC_SHARD_RESB];

static void shard_reset_all(void)
{
    shard_nt = 0;
    shard_used = 0;
    shard_ready = 0;
    shard_res_n = 0;
    shard_res_tok = 0;
    memset(&shard_ctx, 0, sizeof shard_ctx);
    memset(shard_t, 0, sizeof shard_t);
}

/* Краткое описание для PONG/HELLO: что узел держит и сколько свободно. */
static void shard_describe(char *out, size_t n)
{
    if (!shard_ready) {
        snprintf(out, n, "свободно %d из %d байт", (int)(DC_SHARD_MAX_BYTES - shard_used),
                 (int)DC_SHARD_MAX_BYTES);
        return;
    }
    const char *st = shard_ctx.stage == LLM_STAGE_EMBED
                       ? "вход"
                       : shard_ctx.stage == LLM_STAGE_HEAD ? "выход" : "слои";
    snprintf(out, n, "%s %d..%d embd %d, занято %d из %d байт", st, shard_ctx.first,
             shard_ctx.last, shard_ctx.n_embd, (int)shard_used, (int)DC_SHARD_MAX_BYTES);
}

/* Находит тензор по имени; возвращает индекс или -1. */
static int shard_find(const char *name)
{
    for (int i = 0; i < shard_nt; i++)
        if (strcmp(shard_t[i].name, name) == 0) return i;
    return -1;
}

/* «wt|имя|ndim|d0,d1,..|тип|off|total|base64» — кусок тензора в пул. */
static int shard_put(const req_t *r, char *out, size_t outsz, char *detail, size_t detailsz,
                     int *ok)
{
    const char *p = r->payload;
    if (strncmp(p, "wt|", 3) != 0) return -1;
    p += 3;

    char name[DC_SHARD_NAME];
    int  len = 0;
    while (*p && *p != '|' && len < (int)sizeof name - 1) name[len++] = *p++;
    name[len] = '\0';
    if (*p != '|' || len == 0) return -1;
    p++;

    /* сначала число измерений, затем сами размеры: «ndim|d0,d1,..» */
    unsigned long dims[4];
    unsigned      ndim = 0;
    {
        const char *e = NULL;
        unsigned long v = strtoul(p, (char **)&e, 10);
        if (e == p || *e != '|' || v == 0 || v > 4) return -1;
        ndim = (unsigned)v;
        p = e + 1;
    }
    for (unsigned i = 0; i < ndim; i++) {
        const char *e = NULL;
        unsigned long v = strtoul(p, (char **)&e, 10);
        if (e == p) return -1;
        dims[i] = v;
        p = e;
        if (i + 1 < ndim) {
            if (*p != ',') return -1;
            p++;
        }
    }
    if (*p != '|') return -1;
    p++;

    const char *e = NULL;
    unsigned type = (unsigned)strtoul(p, (char **)&e, 10);
    if (e == p || *e != '|') return -1;
    p = e + 1;
    size_t off = (size_t)strtoull(p, (char **)&e, 10);
    if (e == p || *e != '|') return -1;
    p = e + 1;
    size_t total = (size_t)strtoull(p, (char **)&e, 10);
    if (e == p || *e != '|') return -1;
    p = e + 1;
    if (!*p) return -1;
    if (type > 1 || total == 0 || ndim == 0) return -1;
    size_t esz = type == 1 ? 2 : 4;
    size_t nbytes = 1;
    for (unsigned i = 0; i < ndim; i++) {
        if (dims[i] == 0) return -1;
        nbytes *= (size_t)dims[i];
    }
    nbytes *= esz;
    if (nbytes != total) return -1;      /* размеры и длина обязаны сойтись */

    int slot = shard_find(name);
    if (slot < 0) {
        if (shard_nt >= DC_SHARD_TENSORS) {
            scpy(out, outsz, "ОШИБКА");
            scpy(detail, detailsz, "узел держит не более ");
            scat_u(detail, detailsz, (unsigned long)DC_SHARD_TENSORS);
            scat(detail, detailsz, " тензоров — модель великовата для одного устройства");
            *ok = 0;
            return 0;
        }
        slot = shard_nt++;
        snprintf(shard_t[slot].name, sizeof shard_t[slot].name, "%s", name);
        shard_t[slot].nbytes = total;
        shard_t[slot].type = type;
        shard_t[slot].n_dims = ndim;
        for (unsigned i = 0; i < ndim; i++) shard_t[slot].dims[i] = dims[i];
        shard_t[slot].off = shard_used;
        off = 0;
        /* выравивание под float: иначе на некоторых платформах будет SIGBUS */
        while (shard_used & 3) shard_used++;
        shard_t[slot].off = shard_used;
    } else if (shard_t[slot].nbytes != total || shard_t[slot].type != type ||
               shard_t[slot].n_dims != ndim) {
        scpy(out, outsz, "ОШИБКА");
        scpy(detail, detailsz, "тензор ");
        scat(detail, detailsz, name);
        scat(detail, detailsz, " пришлён с другими размерами");
        *ok = 0;
        return 0;
    }

    if (shard_t[slot].off + total > sizeof shard_pool) {
        scpy(out, outsz, "ОШИБКА");
        scpy(detail, detailsz, "в узле не хватает памяти: нужно ");
        scat_u(detail, detailsz, (unsigned long)total);
        scat(detail, detailsz, " байт, свободно ");
        scat_u(detail, detailsz, (unsigned long)(sizeof shard_pool - shard_used));
        *ok = 0;
        return 0;
    }

    size_t got = 0;
    if (b64_decode(p, shard_pool + shard_t[slot].off + off, total - off, &got) != 0)
        return -1;
    shard_used += (size_t)got;

    scpy(out, outsz, "тензор ");
    scat(out, outsz, name);
    scat(out, outsz, " принят по ");
    scat_u(out, outsz, (unsigned long)(off + got));
    scat(out, outsz, " из ");
    scat_u(out, outsz, (unsigned long)total);
    *ok = 1;
    return 0;
}

/* «shape|stage|first|last|n_embd|n_head|n_head_kv|n_ff|head_dim|n_vocab|n_layer|eps|rope|n_ctx» */
static int shard_shape(const req_t *r, char *out, size_t outsz, char *detail, size_t detailsz,
                      int *ok)
{
    char *p = r->payload;
    if (strncmp(p, "shape|", 6) != 0) return -1;
    p += 6;
    char *f[15];
    int   nf = 0;
    while (nf < 15 && *p) {
        f[nf++] = p;
        while (*p && *p != '|') p++;
        if (*p == '|') { *p = '\0'; p++; }
    }
    if (nf < 13) return -1;   /* stage|first|last|embd|head|head_kv|ff|hdim|vocab|nlayer|eps|rope|n_ctx */
    int stage = atoi(f[0]), first = atoi(f[1]), last = atoi(f[2]);
    int n_embd = atoi(f[3]), n_head = atoi(f[4]), n_head_kv = atoi(f[5]);
    int n_ff = atoi(f[6]), head_dim = atoi(f[7]), n_vocab = atoi(f[8]);
    int n_layer = atoi(f[9]);
    double eps = atof(f[10]), rope = atof(f[11]);
    int n_ctx = atoi(f[12]);

    /* собираем «GGUF» из тензоров пула */
    shard_g.file = (const unsigned char *)shard_pool;
    shard_g.size = shard_used;
    shard_g.version = 3;
    shard_g.alignment = 32;
    shard_g.n_tensors = (size_t)shard_nt;
    shard_g.tensors = shard_gt;
    shard_g.n_meta = 0;
    shard_g.meta = shard_gm;

    for (int i = 0; i < shard_nt; i++) {
        shard_gt[i].name = shard_t[i].name;
        shard_gt[i].n_dims = shard_t[i].n_dims;
        for (unsigned k = 0; k < shard_t[i].n_dims; k++)
            shard_gt[i].dims[k] = shard_t[i].dims[k];
        shard_gt[i].type = shard_t[i].type;
        shard_gt[i].data = shard_pool + shard_t[i].off;
        shard_gt[i].nbytes = shard_t[i].nbytes;
    }

    shard_gm_n = 0;
#define META_STR(nm, val)                                                    \
    do {                                                                      \
        static const char k[] = nm;                                          \
        shard_gm[shard_gm_n].key = k;                                        \
        shard_gm[shard_gm_n].type = 8;                                       \
        shard_gm[shard_gm_n].str = val;                                      \
        shard_gm_n++;                                                         \
    } while (0)
#define META_U32(nm, val)                                                     \
    do {                                                                      \
        static const char k[] = nm;                                          \
        shard_gm[shard_gm_n].key = k;                                        \
        shard_gm[shard_gm_n].type = 4;                                       \
        shard_gm[shard_gm_n].num = (double)(val);                             \
        shard_gm_n++;                                                         \
    } while (0)
#define META_F32(nm, val)                                                     \
    do {                                                                      \
        static const char k[] = nm;                                          \
        shard_gm[shard_gm_n].key = k;                                        \
        shard_gm[shard_gm_n].type = 6;                                       \
        shard_gm[shard_gm_n].num = (double)(val);                             \
        shard_gm_n++;                                                         \
    } while (0)
    META_STR("general.architecture", "llama");
    META_U32("llama.block_count", (unsigned long)n_layer);
    META_U32("llama.embedding_length", (unsigned long)n_embd);
    META_U32("llama.feed_forward_length", (unsigned long)n_ff);
    META_U32("llama.attention.head_count", (unsigned long)n_head);
    META_U32("llama.attention.head_count_kv", (unsigned long)n_head_kv);
    META_F32("llama.attention.layer_norm_rms_epsilon", (float)eps);
    META_F32("llama.rope.freq_base", (float)rope);
    META_U32("llama.rope.dimension_count", (unsigned long)head_dim);
    META_U32("llama.context_length", (unsigned long)n_ctx);
    META_U32("llama.vocab_size", (unsigned long)n_vocab);
    shard_g.n_meta = (size_t)shard_gm_n;
#undef META_STR
#undef META_U32
#undef META_F32

    shard_ready = 0;
    char err[512] = {0};
    if (llm_ctx_open(&shard_ctx, &shard_g, stage, first, last, n_ctx, shard_layers,
                     shard_arena, DC_SHARD_ARENA, err, sizeof err) != 0) {
        /* разбираем, каких тензоров не хватило, — планировщику это важно */
        char miss[384];
        miss[0] = '\0';
        snprintf(miss, sizeof miss, "%s", err);
        scpy(out, outsz, "ОШИБКА");
        scpy(detail, detailsz, miss);
        *ok = 0;
        return 0;
    }
    shard_res_n = n_embd;
    shard_ready = 1;
    llm_reset(&shard_ctx);

    const char *st = stage == LLM_STAGE_EMBED ? "вход"
                       : stage == LLM_STAGE_HEAD ? "выход" : "слои";
    scpy(out, outsz, st);
    scat(out, outsz, " ");
    scat_u(out, outsz, (unsigned long)first);
    scat(out, outsz, "..");
    scat_u(out, outsz, (unsigned long)last);
    scat(out, outsz, ", embd ");
    scat_u(out, outsz, (unsigned long)n_embd);
    scat(out, outsz, ", свободно ");
    scat_u(out, outsz, (unsigned long)(sizeof shard_pool - shard_used));
    *ok = 1;
    return 0;
}

/* «gen|токен» — один шаг; результат кэшируется для part. */
static int shard_gen(const req_t *r, char *out, size_t outsz, char *detail, size_t detailsz,
                     int *ok)
{
    if (!shard_ready) {
        scpy(out, outsz, "ОШИБКА");
        scpy(detail, detailsz, "стадия не настроена: сначала пришлите shape");
        *ok = 0;
        return 0;
    }
    const char *p = r->payload;
    if (strncmp(p, "gen|", 4) != 0) return -1;
    p += 4;
    const char *e = NULL;
    int tok = (int)strtol(p, (char **)&e, 10);
    if (e == p) return -1;

    /* состояние от предыдущей стадии: gen|токен|base64. У входа его нет,
     * у слоёв и выхода — есть: движок читает вход прямо из shard_res. */
    if (*e == '|' && e[1]) {
        size_t need = (size_t)shard_ctx.n_embd * sizeof(float);
        if (need > sizeof shard_res) {
            scpy(out, outsz, "ОШИБКА");
            scpy(detail, detailsz, "состояние не влезает в буфер узла");
            *ok = 0;
            return 0;
        }
        size_t dec = 0;
        if (b64_decode(e + 1, shard_res, need, &dec) != 0 || dec != need) {
            scpy(out, outsz, "ОШИБКА");
            scpy(detail, detailsz, "не разобрано состояние от предыдущей стадии");
            *ok = 0;
            return 0;
        }
    }

    if (llm_forward(&shard_ctx, tok, shard_res) != 0) {
        scpy(out, outsz, "ОШИБКА");
        scpy(detail, detailsz, "шаг не выполнен: контекст исчерпан или токен вне словаря");
        *ok = 0;
        return 0;
    }
    shard_res_n = shard_ctx.n_embd;
    shard_res_tok = (shard_ctx.stage == LLM_STAGE_HEAD) ? llm_pick(&shard_ctx) : 0;

    if (shard_ctx.stage == LLM_STAGE_HEAD) {
        scpy(out, outsz, "токен ");
        scat_u(out, outsz, (unsigned long)shard_res_tok);
    } else {
        scpy(out, outsz, "ok ");
        scat_u(out, outsz, (unsigned long)shard_res_n);
    }
    scpy(detail, detailsz, "стадия выполнена");
    *ok = 1;
    return 0;
}

/* «part|i|n» — i-й кусок кэшированного результата по n байт. */
static int shard_part(const req_t *r, char *out, size_t outsz, int *ok)
{
    const char *p = r->payload;
    if (strncmp(p, "part|", 5) != 0) return -1;
    p += 5;
    const char *e = NULL;
    long i = strtol(p, (char **)&e, 10);
    if (e == p || *e != '|') return -1;
    p = e + 1;
    long n = strtol(p, (char **)&e, 10);
    if (e == p || i < 0 || n <= 0 || n > (long)sizeof shard_pool) return -1;
    size_t off = (size_t)i;                 /* i — байтовое смещение в состоянии */
    if (off + (size_t)n > (size_t)shard_res_n * sizeof(float)) {
        scpy(out, outsz, "ОШИБКА");
        *ok = 0;
        return 0;
    }
    char b64[b64_encode_size((size_t)n)];
    int  len = b64_encode((const unsigned char *)shard_res + off, (size_t)n, b64, sizeof b64);
    if (len < 0) return -1;
    scpy(out, outsz, b64);
    *ok = 1;
    return 0;
}

static void role_shard(const req_t *r, char *out, size_t outsz, char *detail, size_t detailsz,
                       int *ok)
{
    if (strncmp(r->payload, "info", 4) == 0) {
        char buf[128];
        shard_describe(buf, sizeof buf);
        scpy(out, outsz, buf);
        scpy(detail, detailsz, "");
        *ok = 1;
        return;
    }
    if (strncmp(r->payload, "free", 4) == 0) {
        int had = shard_nt;
        shard_reset_all();
        scpy(out, outsz, "освобождено тензоров ");
        scat_u(out, outsz, (unsigned long)had);
        scpy(detail, detailsz, "");
        *ok = 1;
        return;
    }
    if (strncmp(r->payload, "reset", 5) == 0) {
        llm_reset(&shard_ctx);
        shard_res_n = 0;
        scpy(out, outsz, "сброшено");
        scpy(detail, detailsz, "");
        *ok = 1;
        return;
    }
    if (strncmp(r->payload, "wt|", 3) == 0) {
        if (shard_put(r, out, outsz, detail, detailsz, ok) != 0) {
            scpy(out, outsz, "ОШИБКА");
            scpy(detail, detailsz, "не разобрана команда wt|…");
            *ok = 0;
        }
        return;
    }
    if (strncmp(r->payload, "shape|", 6) == 0) {
        if (shard_shape(r, out, outsz, detail, detailsz, ok) != 0) {
            scpy(out, outsz, "ОШИБКА");
            scpy(detail, detailsz, "не разобрана команда shape|…");
            *ok = 0;
        }
        return;
    }
    if (strncmp(r->payload, "gen|", 4) == 0) {
        if (shard_gen(r, out, outsz, detail, detailsz, ok) != 0) {
            scpy(out, outsz, "ОШИБКА");
            scpy(detail, detailsz, "не разобрана команда gen|…");
            *ok = 0;
        }
        return;
    }
    if (shard_part(r, out, outsz, ok) == 0) return;

    scpy(out, outsz, "ОШИБКА");
    scpy(detail, detailsz,
         "неизвестная команда узлу llm (info | free | reset | wt|… | shape|… | gen|… | part|…)");
    *ok = 0;
}

/* -------------------------------------------------------- ответ ---- */

static void send_reply(int fd, const struct sockaddr_in *to, const char *json)
{
    unsigned char *d = NULL;
    size_t         len = 0;
    if (udp_pack(DC_MSG_RESPONSE, json, &d, &len) == 0) {
        sendto(fd, d, len, 0, (const struct sockaddr *)to, sizeof *to);
        free(d);
    }
}

/* Ошибка до разбора запроса/лимита — тот же формаT, что у ролей:
 * планировщик читает ok/error/max_body/node в любом ответе. */
static void send_error(int fd, const struct sockaddr_in *to, const char *msg)
{
    buf_t b;
    binit(&b, reply_buf, sizeof reply_buf);
    badd(&b, "{\"ok\":false,\"error\":");
    badd_json(&b, msg);
    badd(&b, ",\"max_body\":");
    badd_u(&b, (unsigned long)N.max_body);
    badd(&b, ",\"node\":");
    badd_json(&b, N.tag);
    badd(&b, "}");
    send_reply(fd, to, reply_buf);
}

/* PING -> PONG: планировщик запоминает max_body и больше не присылает
 * этому узлу слишком тяжёлые шаги. */
static void send_pong(int fd, const struct sockaddr_in *to)
{
    buf_t b;
    binit(&b, reply_buf, sizeof reply_buf);
    badd(&b, "{\"ok\":true,\"node\":");
    badd_json(&b, N.tag);
    badd(&b, ",\"arch\":");
    badd_json(&b, DC_ARCH);
    badd(&b, ",\"max_body\":");
    badd_u(&b, (unsigned long)N.max_body);
    /* сколько памяти отдаём под веса модели и что уже держим */
    badd(&b, ",\"free_mem\":");
    badd_u(&b, (unsigned long)N.free_mem);
    badd(&b, ",\"shard\":");
    {
        char sdesc[96];
        shard_describe(sdesc, sizeof sdesc);
        badd_json(&b, sdesc);
    }
    badd(&b, ",\"roles\":\"");
    for (int i = 0; i < N.nroles; i++) {
        if (i) badd(&b, ",");
        badd(&b, role_name(N.serve[i]));
    }
    badd(&b, "\"}");
    send_reply(fd, to, reply_buf);
}

static void build_reply(const req_t *r, buf_t *res)
{
    int idx = role_index(r->role);
    int allowed = role_allowed(idx);

    char out[512], detail[512];
    out[0] = detail[0] = '\0';
    int ok = 1;

    if (idx < 0) {
        scpy(out, sizeof out, "ОШИБКА");
        scpy(detail, sizeof detail, "неизвестная роль");
        ok = 0;
    } else if (!allowed) {
        scpy(out, sizeof out, "ОШИБКА");
        scpy(detail, sizeof detail, "узел ");
        scat(detail, sizeof detail, N.tag);
        scat(detail, sizeof detail, " обслуживает только ");
        for (int i = 0; i < N.nroles; i++) {
            if (i) scat(detail, sizeof detail, ",");
            scat(detail, sizeof detail, role_name(N.serve[i]));
        }
        ok = 0;
    } else if (idx == 0) {
        role_compute(r, out, sizeof out, detail, sizeof detail, &ok);
    } else if (idx == 1) {
        role_verify(r, out, sizeof out, detail, sizeof detail, &ok);
    } else if (idx == 2) {
        role_memorize(r, out, sizeof out, detail, sizeof detail, &ok);
    } else {
        role_shard(r, out, sizeof out, detail, sizeof detail, &ok);
    }

    binit(res, reply_buf, sizeof reply_buf);
    badd(res, "{\"ok\":");
    badd(res, ok ? "true" : "false");
    badd(res, ",\"output\":");
    badd_json(res, out);
    badd(res, ",\"detail\":");
    badd_json(res, detail);
    badd(res, ",\"node\":");
    badd_json(res, N.tag);
    badd(res, ",\"max_body\":");
    badd_u(res, (unsigned long)N.max_body);
    badd(res, ",\"free_mem\":");
    badd_u(res, (unsigned long)N.free_mem);
    if (role_allowed(3)) {
        char sdesc[96];
        shard_describe(sdesc, sizeof sdesc);
        badd(res, ",\"shard\":");
        badd_json(res, sdesc);
    }
    badd(res, ",\"roles\":\"");
    for (int i = 0; i < N.nroles; i++) {
        if (i) badd(res, ",");
        badd(res, role_name(N.serve[i]));
    }
    badd(res, "\"");
    if (!ok) {
        badd(res, ",\"error\":");
        badd_json(res, detail[0] ? detail : "проверка не пройдена");
    }
    badd(res, "}");
}

/* -------------------------------------------------------- объявление ---- */
/* Узел объявляет себя постоянно, а не только при старте: планировщик
 * может подняться позже. Пакет 60–80 байт раз в 10 с — примерно ничего.
 *
 * Но объявление по таймеру бесполезно для того, кто ищет узлы прямо сейчас:
 * к моменту `dcctl discover` последнее объявление уже прошло, а следующее
 * будет через десять секунд. Поэтому узел ещё и слушает порт объявлений и
 * отвечает на PING немедленным HELLO — искать можно по запросу. */

static void broadcast_hello(void)
{
    char list[96] = "";
    for (int i = 0; i < N.nroles; i++) {
        if (i) scat(list, sizeof list, ",");
        scat(list, sizeof list, role_name(N.serve[i]));
    }
    char sdesc[96];
    shard_describe(sdesc, sizeof sdesc);
    unsigned char *d = NULL;
    size_t         dl = 0;
    if (udp_pack_hello(N.max_body, list, DC_ARCH, N.tag, N.port, N.free_mem, sdesc, &d, &dl) != 0)
        return;

    /* из эфемерного сокета: адрес источника объявления ничего не значит,
     * важно лишь поле port внутри пакета */
    int s = udp_socket_client(NULL, 0);
    if (s < 0) {
        free(d);
        return;
    }
    int on = 1;
    setsockopt(s, SOL_SOCKET, SO_BROADCAST, &on, sizeof on);
    struct sockaddr_in baddr;
    memset(&baddr, 0, sizeof baddr);
    baddr.sin_family = AF_INET;
    baddr.sin_port = htons(DC_DISCOVERY_PORT);
    baddr.sin_addr.s_addr = htonl(INADDR_BROADCAST);
    sendto(s, d, dl, 0, (struct sockaddr *)&baddr, sizeof baddr);

    /* 127.255.255.255 — широковещание только по loopback: в контейнере и на
     * машине без настроенного broadcast этот адрес доходит до своих же
     * слушателей, а 255.255.255.255 может не дойти */
    baddr.sin_addr.s_addr = htonl(0x7FFFFFFF);
    sendto(s, d, dl, 0, (struct sockaddr *)&baddr, sizeof baddr);

    close(s);
    free(d);
}

/* -------------------------------------------------------- запуск ---- */

static const char *prog_name(char **argv)
{
    const char *slash = strrchr(argv[0], '/');
    return slash ? slash + 1 : argv[0];
}

static void usage(const char *prog)
{
    dprint("%s — узел распределённых вычислений (UDP)\n", prog);
    dprint("\n");
    dprint("  %s --role compute [--role verify] [--role memorize] [--role shard] --port N\n",
           prog);
    dprint("      [--tag ИМЯ] [--state DIR] [--max-body Б] [--free-mem Б] [--hello-interval N]\n");
    dprint("\n");
    dprint("Один процесс может обслуживать несколько ролей — для устройств,\n");
    dprint("где запуск трёх процессов не помещается в память.\n");
    dprint("\n");
    dprint("  --max-body   сколько байт тела узел примет (планировщик не пришлёт\n");
    dprint("               больше; по умолчанию %d)\n", DC_MAX_BODY);
    dprint("  --free-mem   сколько байт весов модели узел готов держать: по этому\n");
    dprint("               числу планировщик решает, какие слои положить сюда\n");
    dprint("               (по умолчанию %d, на Linux — свободная память)\n",
           DC_SHARD_MAX_BYTES);
    dprint("  --hello-interval N  как часто объявлять себя по широковещанию,\n");
    dprint("               секунд (0 = не объявлять; по умолчанию %d)\n", DC_HELLO_SECS);
    dprint("\n");
    dprint("Сборка lite (make lite) — маленькие буферы для SAMD51/RP2040/ESP32.\n");
}

/* Сколько памяти отдать под веса. На большой машине честнее сказать
 * «свободная память», но не больше размера пула; на микроконтроллере
 * /proc нет, и там берётся весь пул. */
static long default_free_mem(void)
{
#if defined(__linux__) && !defined(DC_LITE)
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[160];
        while (fgets(line, sizeof line, f)) {
            long kb = 0;
            if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1 && kb > 0) {
                fclose(f);
                long avail = kb * 1024;
                return avail < (long)DC_SHARD_MAX_BYTES ? avail : (long)DC_SHARD_MAX_BYTES;
            }
        }
        fclose(f);
    }
#endif
    return (long)DC_SHARD_MAX_BYTES;
}

int main(int argc, char **argv)
{
    signal(SIGPIPE, SIG_IGN);

    int no_hello = 0;
    int hello_secs = DC_HELLO_SECS;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--role") == 0 && i + 1 < argc) {
            int idx = role_index(argv[++i]);
            if (idx < 0) {
                dprint("неизвестная роль: %s\n", argv[i]);
                return 2;
            }
            int dup = 0;
            for (int k = 0; k < N.nroles; k++)
                if (N.serve[k] == idx) dup = 1;
            if (!dup && N.nroles < 4) N.serve[N.nroles++] = idx;
        } else if (strcmp(a, "--port") == 0 && i + 1 < argc) {
            N.port = atoi(argv[++i]);
        } else if (strcmp(a, "--tag") == 0 && i + 1 < argc) {
            scpy(N.tag, sizeof N.tag, argv[++i]);
        } else if (strcmp(a, "--state") == 0 && i + 1 < argc) {
            scpy(N.state_dir, sizeof N.state_dir, argv[++i]);
        } else if (strcmp(a, "--max-body") == 0 && i + 1 < argc) {
            int mb = atoi(argv[++i]);
            if (mb > 0 && mb < DC_MAX_BODY) N.max_body = mb;
        } else if (strcmp(a, "--free-mem") == 0 && i + 1 < argc) {
            long fm = atol(argv[++i]);
            if (fm > 0 && fm < DC_SHARD_MAX_BYTES) N.free_mem = fm;
        } else if (strcmp(a, "--hello-interval") == 0 && i + 1 < argc) {
            hello_secs = atoi(argv[++i]);
            if (hello_secs <= 0) no_hello = 1;
        } else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            usage(prog_name(argv));
            return 0;
        } else {
            dprint("неизвестный аргумент: %s\n", a);
            return 2;
        }
    }

    if (!N.nroles || !N.port) {
        usage(prog_name(argv));
        return 2;
    }
    if (!N.tag[0]) scpy(N.tag, sizeof N.tag, "node");
    if (!N.state_dir[0]) scpy(N.state_dir, sizeof N.state_dir, "state");
    if (N.max_body <= 0) N.max_body = DC_MAX_BODY;
    if (N.free_mem <= 0) N.free_mem = default_free_mem();
    scpy(mpath, sizeof mpath, N.state_dir);
    scat(mpath, sizeof mpath, "/memory.log");
    if (role_allowed(2)) mem_count_init();
    mkdir(N.state_dir, 0775);

    char err[192] = {0};
    int  fd = udp_socket_bound("0.0.0.0", N.port, err, sizeof err);
    if (fd < 0) {
        dprint("%s\n", err);
        return 1;
    }

    /* порт объявлений: отвечаем здесь на «кто тут?» — см. broadcast_hello().
     * Буферы маленькие (4 КБ): на этом сокете ходят только пакеты по 60–80
     * байт, а мегабайт приёма на микроконтроллере — непозволительная роскошь.
     * Если порт занят (например, забытый процесс) — работаем без него. */
    int disco_fd = -1;
    if (!no_hello) {
        char derr[192] = {0};
        disco_fd = udp_socket_bound_ex("0.0.0.0", DC_DISCOVERY_PORT, 4096, 4096, derr, sizeof derr);
        if (disco_fd < 0) dprint("объявления по сети отключены: %s\n", derr);
    }

    const char *prog = prog_name(argv);
    dprint("%s: порт %d, роли", prog, N.port);
    for (int i = 0; i < N.nroles; i++) dprint(" %s", role_name(N.serve[i]));
    dprint(", лимит тела %d байт, платформа %s%s", N.max_body, DC_ARCH,
           no_hello ? ", без объявления" : "");
    if (role_allowed(3)) {
        char sdesc[96];
        shard_describe(sdesc, sizeof sdesc);
        dprint(", под модель %d байт", (int)N.free_mem);
        dprint(" (%s)", sdesc);
    }
    dprint("\n");

    /*
     * Объявления идут из основного цикла через poll(), а не отдельным
     * sleep: иначе узел первые секунды не отвечает на запросы вообще.
     */
    long hello_every_ms = no_hello ? 0 : (long)hello_secs * 1000;
    long next_hello = 0;
    long last_hello_asked = 0;   /* защита от лавины «кто тут?» */

    for (;;) {
        if (next_hello == 0) {
            broadcast_hello();
            next_hello = now_ms() + hello_every_ms;
        }

        int wait_ms = (int)(next_hello - now_ms());
        if (wait_ms < 0 || wait_ms > 500) wait_ms = 500;

        struct pollfd pfd[2] = {
            { .fd = fd, .events = POLLIN, .revents = 0 },
            { .fd = disco_fd, .events = POLLIN, .revents = 0 },
        };
        int nready = poll(pfd, disco_fd >= 0 ? 2 : 1, wait_ms);
        if (nready < 0) {
            if (errno == EINTR) continue;
            continue;
        }
        if (nready == 0) {
            if (hello_every_ms > 0 && now_ms() >= next_hello) next_hello = 0;
            continue;
        }

        /* приёмный буфер в BSS: на устройстве с 32 КБ ОЗУ стек такого
         * размера — это треть всей памяти */
        static unsigned char buf[DC_UDP_MAX_DGRAM + 16];
        struct sockaddr_in from;
        socklen_t          flen = sizeof from;

        /* сперва разберёмся с портом объявлений: там лежат только запросы
         * «кто тут?», и на них нужно ответить, а не работать */
        if (disco_fd >= 0 && (pfd[1].revents & POLLIN)) {
            ssize_t dr = recvfrom(disco_fd, buf, sizeof buf, 0, (struct sockaddr *)&from, &flen);
            dc_msg  dm;
            char    dperr[192] = {0};
            if (dr > 0 && udp_unpack(buf, (size_t)dr, &dm, dperr, sizeof dperr) == 0 &&
                dm.type == DC_MSG_PING) {
                /* не чаще раза в секунду: иначе десяток узлов в сети
                 * размножит объявления на запрос каждого */
                long now = now_ms();
                if (now - last_hello_asked >= 1000) {
                    last_hello_asked = now;
                    broadcast_hello();
                }
            }
            continue;
        }
        if (!(pfd[0].revents & POLLIN)) continue;

        flen = sizeof from;
        ssize_t r = recvfrom(fd, buf, sizeof buf, 0, (struct sockaddr *)&from, &flen);
        if (r < 0) continue;

        dc_msg m;
        char   perr[192] = {0};
        if (udp_unpack(buf, (size_t)r, &m, perr, sizeof perr) != 0) continue;
        if (m.type == DC_MSG_HELLO) continue;
        if (m.type == DC_MSG_PING) {
            send_pong(fd, &from);
            continue;
        }
        if (m.type != DC_MSG_REQUEST) continue;

        /* тело крупнее лимита узла — отвечаем отказом, а не молчим:
         * молчание выглядело бы как потеря пакета и съедало бы таймаут. */
        if ((int)m.body_len > N.max_body) {
            scpy(perr, sizeof perr, "задание ");
            scat_u(perr, sizeof perr, (unsigned long)m.body_len);
            scat(perr, sizeof perr, " байт больше лимита узла ");
            scat_u(perr, sizeof perr, (unsigned long)N.max_body);
            send_error(fd, &from, perr);
            continue;
        }

        size_t len = m.body_len < sizeof rx_buf - 1 ? m.body_len : sizeof rx_buf - 1;
        memcpy(rx_buf, m.body, len);
        rx_buf[len] = '\0';

        /* разобранный запрос тоже в BSS: с payload на 60 КБ он занял бы
         * больше, чем весь стек процесса на MCU */
        static req_t req;
        if (!req_parse(rx_buf, &req)) {
            send_error(fd, &from, "запрос не JSON или без поля role");
            continue;
        }

        N.served++;
        buf_t res;
        build_reply(&req, &res);
        dprint("[%s/%s] %s -> %s\n", req.role[0] ? req.role : "?", N.tag,
               req.title[0] ? req.title : "?", res.p ? res.p : "");
        send_reply(fd, &from, res.p);
    }
}
