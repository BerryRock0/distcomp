/*
 * Юнит-тесты вычислителя выражений.
 * Собирается как bin/expr_test, запускается из scripts/test.sh.
 *
 * Здесь намеренно проверяются случаи, которые однажды молча ломались:
 *   max(3,9)   — запятая проглатывалась как разделитель тысяч
 *   log10(1000)— идентификатор читался только по буквам, давало «log»
 *   1_000_000  — подчёркивание разрывало число на две части
 */

#include "../src/expr.h"

#include <stdio.h>
#include <string.h>

static int failed = 0;
static int total = 0;

static void expect(const char *expr, double want)
{
    total++;
    double v = 0;
    char   err[200] = {0};
    int    rc = expr_eval(expr, &v, err, sizeof err);
    if (rc != 0) {
        printf("  УПАЛ  %-34s ошибка: %s\n", expr, err);
        failed++;
        return;
    }
    double eps = 1e-9 * (want < 0 ? -want : want);
    if (eps < 1e-9) eps = 1e-9;
    if (v - want > eps || want - v > eps) {
        printf("  УПАЛ  %-34s получено %.10g, ждали %.10g\n", expr, v, want);
        failed++;
    }
}

static void expect_err(const char *expr)
{
    total++;
    double v = 0;
    char   err[200] = {0};
    if (expr_eval(expr, &v, err, sizeof err) == 0) {
        printf("  УПАЛ  %-34s ожидалась ошибка, получено %.10g\n", expr, v);
        failed++;
    } else if (!err[0]) {
        printf("  УПАЛ  %-34s ошибка без описания\n", expr);
        failed++;
    }
}

int main(void)
{
    /* арифметика и приоритеты */
    expect("2+3*4", 14);
    expect("(2+3)*4", 20);
    expect("2-10/4", -0.5);
    expect("-5 + 3", -2);
    expect("--5", 5);
    expect("17 % 5", 2);
    expect("2^10", 1024);
    expect("2^3^2", 512);            /* правоассоциативно */
    expect("(2^3)^2", 64);
    expect("-2^2", -4);               /* минус слабее степени */
    expect("1/3", 1.0 / 3.0);
    expect("((2+3)*(4-1))/5", 3);

    /* числа: подчёркивания, экспонента, дробные */
    expect("1_000_000 / 4", 250000);
    expect("1_2_3", 123);
    expect("1.5e3 + 2e-2", 1500.02);
    expect("1.5E3", 1500);
    expect("2.5e+2", 250);
    expect(".5 + .5", 1);
    expect("7.", 7);

    /* функции одного аргумента */
    expect("sqrt(144)", 12);
    expect("abs(-7.5)", 7.5);
    expect("round(2.5)+floor(2.9)+ceil(2.1)", 8);
    expect("exp(1)", 2.718281828459045);
    expect("sign(-9)", -1);
    expect("log2(8)", 3);
    expect("log10(1000)", 3);

    /* функции двух аргументов — здесь были поломки */
    expect("max(3,9)", 9);
    expect("min(4,2)", 2);
    expect("pow(2,10)", 1024);
    expect("atan2(1,1)", 0.7853981633974483);
    expect("max(1,2) + min(3,4)", 5);

    /* константы */
    expect("pi", 3.14159265358979323846);
    expect("e", 2.71828182845904523536);
    expect("tau", 6.28318530717958647692);

    /* регистр не важен */
    expect("SQRT(144) + MAX(1,2)", 14);

    /* ошибки */
    expect_err("1/0");
    expect_err("17 % 0");
    expect_err("(2+3");
    expect_err("2 3");
    expect_err("sqrt()");
    expect_err("unknownfn(2)");
    expect_err("max(1,2,3)");        /* max принимает ровно два аргумента */
    expect_err("pow(2)");
    expect_err("1..2");

    printf("вычислитель: %d проверок, упало %d\n", total, failed);
    return failed == 0 ? 0 : 1;
}
