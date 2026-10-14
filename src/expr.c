#include "expr.h"
#include "util.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p;
    char       *err;
    size_t      errlen;
    int         failed;
} ectx;

static void efail(ectx *c, const char *msg)
{
    if (!c->failed && c->err && c->errlen) snprintf(c->err, c->errlen, "%s", msg);
    c->failed = 1;
}

/* Пробелы и переводы строк пропускаются. Запятая сюда намеренно НЕ входит:
 * это разделитель аргументов функции (max(1,2)), а не разделитель тысяч. */
static void eskip(ectx *c)
{
    while (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r') c->p++;
}

static double eexpr(ectx *c);

static int efactor(ectx *c, char op, double a, double b, double *out)
{
    switch (op) {
    case '+': *out = a + b; return 0;
    case '-': *out = a - b; return 0;
    case '*': *out = a * b; return 0;
    case '/':
        if (fabs(b) < 1e-15) {
            efail(c, "деление на ноль");
            return -1;
        }
        *out = a / b;
        return 0;
    case '%':
        if (fabs(b) < 1e-15) {
            efail(c, "остаток от деления на ноль");
            return -1;
        }
        *out = fmod(a, b);
        return 0;
    default:
        efail(c, "неизвестный оператор");
        return -1;
    }
}

static double eprimary(ectx *c)
{
    eskip(c);
    if (c->failed) return 0;

    if (*c->p == '(') {
        c->p++;
        double v = eexpr(c);
        if (c->failed) return 0;
        eskip(c);
        if (*c->p == ')') {
            c->p++;
            return v;
        }
        efail(c, "не закрыта скобка");
        return 0;
    }

    if (isdigit((unsigned char)*c->p) || *c->p == '.') {
        /* Число с необязательными подчёркиваниями-разделителями: 1_000_000.
         * Собираем токен вручную, потому что strtod останавливается на '_',
         * а пропуск его через eskip разрывал число на «1» и «000000». */
        const char *start = c->p;
        while (isdigit((unsigned char)*c->p) || *c->p == '_' || *c->p == '.') c->p++;
        /* экспонента: e/E, затем знак и цифры — тоже с подчёркиваниями */
        if ((*c->p == 'e' || *c->p == 'E') &&
            (isdigit((unsigned char)c->p[1]) ||
             ((c->p[1] == '+' || c->p[1] == '-') && isdigit((unsigned char)c->p[2])))) {
            c->p++;
            if (*c->p == '+' || *c->p == '-') c->p++;
            while (isdigit((unsigned char)*c->p) || *c->p == '_') c->p++;
        }

        char   numbuf[128];
        /* длинный литерал нельзя обрезать: strtod молча принял бы обрезок
         * как другое число (1 и 200 нулей превратились бы в 1 и 126 нулей) */
        size_t ndig = 0;
        for (const char *q = start; q < c->p; q++)
            if (*q != '_') ndig++;
        if (ndig >= sizeof numbuf) {
            c->p = start;
            efail(c, "число слишком длинное");
            return 0;
        }

        size_t k = 0;
        for (const char *q = start; q < c->p; q++)
            if (*q != '_') numbuf[k++] = *q;
        numbuf[k] = '\0';

        char  *end = NULL;
        double v = strtod(numbuf, &end);
        if (end == numbuf || (end && *end != '\0')) {
            c->p = start;
            efail(c, "ожидалось число");
            return 0;
        }
        return v;
    }

    /* функции и константы */
    static const struct {
        const char       *name;
        int               arity;
        double          (*fn1)(double);
        double          (*fn2)(double, double);
    } fns[] = {
        { "sqrt", 1, sqrt, NULL },   { "abs", 1, fabs, NULL },
        { "floor", 1, floor, NULL }, { "ceil", 1, ceil, NULL },
        { "round", 1, round, NULL }, { "sin", 1, sin, NULL },
        { "cos", 1, cos, NULL },     { "tan", 1, tan, NULL },
        { "asin", 1, asin, NULL },   { "acos", 1, acos, NULL },
        { "atan", 1, atan, NULL },   { "exp", 1, exp, NULL },
        { "log2", 1, log2, NULL },   { "log10", 1, log10, NULL },
        { "pow", 2, NULL, pow },     { "atan2", 2, NULL, atan2 },
        { "min", 2, NULL, NULL },    { "max", 2, NULL, NULL },
        { "sign", 1, NULL, NULL },
    };

    const char *save = c->p;
    eskip(c);
    const char *idstart = c->p;
    /* идентификатор: буква, дальше буквы и цифры — иначе log10 превратится в log */
    if (isalpha((unsigned char)*c->p)) {
        c->p++;
        while (isalnum((unsigned char)*c->p) || *c->p == '_') c->p++;
    }
    size_t idlen = (size_t)(c->p - idstart);

    if (idlen == 0) {
        c->p = save;
        efail(c, "неожиданный символ");
        return 0;
    }

    char name[16];
    size_t n = idlen < sizeof name - 1 ? idlen : sizeof name - 1;
    for (size_t i = 0; i < n; i++) name[i] = (char)tolower((unsigned char)idstart[i]);
    name[n] = '\0';

    if (strcmp(name, "pi") == 0) return M_PI;
    if (strcmp(name, "e") == 0) return M_E;
    if (strcmp(name, "tau") == 0) return 2.0 * M_PI;

    eskip(c);
    if (*c->p != '(') {
        efail(c, "неизвестный идентификатор");
        return 0;
    }
    c->p++;

    double args[4] = { 0, 0, 0, 0 };
    int    argc = 0;
    for (;;) {
        eskip(c);
        if (*c->p == ')') {
            c->p++;
            break;
        }
        if (argc >= 4) {
            efail(c, "слишком много аргументов");
            return 0;
        }
        args[argc++] = eexpr(c);
        if (c->failed) return 0;
        eskip(c);
        if (*c->p == ',') {
            c->p++;
            continue;
        }
        if (*c->p == ')') {
            c->p++;
            break;
        }
        efail(c, "ожидалась ',' или ')'");
        return 0;
    }

    for (size_t i = 0; i < sizeof fns / sizeof fns[0]; i++) {
        if (strcmp(fns[i].name, name) != 0) continue;
        if (argc != fns[i].arity) {
            efail(c, "неверное число аргументов функции");
            return 0;
        }
        if (fns[i].fn1) return fns[i].fn1(args[0]);
        if (fns[i].fn2) return fns[i].fn2(args[0], args[1]);
        if (strcmp(name, "sign") == 0) return (args[0] > 0) - (args[0] < 0);
        if (strcmp(name, "min") == 0) return args[0] < args[1] ? args[0] : args[1];
        if (strcmp(name, "max") == 0) return args[0] > args[1] ? args[0] : args[1];
    }

    efail(c, "неизвестная функция");
    return 0;
}

/* Унарный знак и возведение в степень. Знак обрабатывается здесь, а не в
 * eprimary, иначе -2^2 посчиталось бы как (-2)^2 = 4. В математике и в
 * Python -2^2 = -4: знак слабее степени. */
static double eunary(ectx *c)
{
    eskip(c);
    if (*c->p == '+' || *c->p == '-') {
        char   sign = *c->p;
        c->p++;
        double v = eunary(c);          /* знак распространяется на всю степень */
        if (c->failed) return 0;
        return sign == '-' ? -v : v;
    }

    double a = eprimary(c);
    if (c->failed) return 0;
    eskip(c);
    /* правоассоциативное возведение в степень */
    if (*c->p == '^') {
        c->p++;
        double b = eunary(c);
        if (c->failed) return 0;
        return pow(a, b);
    }
    return a;
}

static double eterm(ectx *c)
{
    double a = eunary(c);
    if (c->failed) return 0;
    for (;;) {
        eskip(c);
        char op = *c->p;
        if (op != '*' && op != '/' && op != '%') break;
        c->p++;
        double b = eunary(c);
        if (c->failed) return 0;
        double r = 0;
        if (efactor(c, op, a, b, &r) != 0) return 0;
        a = r;
    }
    return a;
}

static double eexpr(ectx *c)
{
    double a = eterm(c);
    if (c->failed) return 0;
    for (;;) {
        eskip(c);
        char op = *c->p;
        if (op != '+' && op != '-') break;
        c->p++;
        double b = eterm(c);
        if (c->failed) return 0;
        double r = 0;
        if (efactor(c, op, a, b, &r) != 0) return 0;
        a = r;
    }
    return a;
}

int expr_eval(const char *s, double *out, char *err, size_t errlen)
{
    if (err && errlen) err[0] = '\0';
    if (!s || !*s) {
        if (err && errlen) snprintf(err, errlen, "пустое выражение");
        return -1;
    }
    ectx c = { .p = s, .err = err, .errlen = errlen, .failed = 0 };
    double v = eexpr(&c);
    if (c.failed) return -1;
    eskip(&c);
    if (*c.p != '\0') {
        snprintf(err, errlen, "лишние символы после выражения: «%s»", c.p);
        return -1;
    }
    if (!isfinite(v)) {
        snprintf(err, errlen, "результат не является конечным числом");
        return -1;
    }
    if (out) *out = v;
    return 0;
}

int expr_looks_numeric(const char *s)
{
    int digit = 0, op = 0;
    for (const char *q = s; q && *q; q++) {
        if (isdigit((unsigned char)*q)) digit = 1;
        else if (*q == '+' || *q == '-' || *q == '*' || *q == '/' || *q == '^') op = 1;
    }
    return digit && op;
}

void expr_format(double v, char *buf, size_t n)
{
    if (fabs(v - floor(v)) < 1e-9 && fabs(v) < 1e15)
        snprintf(buf, n, "%.0f", v);
    else {
        snprintf(buf, n, "%.10g", v);
    }
}
