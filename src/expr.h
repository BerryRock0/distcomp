#ifndef DC_EXPR_H
#define DC_EXPR_H

#include <stddef.h>

/* Вычисляет арифметическое выражение. Поддерживает + - * / % ^ (),
 * унарный минус, функции sqrt abs min max pow floor ceil round sin cos tan
 * asin acos atan exp log log2 log10 и константы pi, e.
 * Возвращает 0 при успехе, -1 при ошибке (err заполняется). */
int expr_eval(const char *s, double *out, char *err, size_t errlen);

/* Грубая проверка «похоже ли на арифметику»: есть цифра и оператор. */
int expr_looks_numeric(const char *s);

/* Красивое форматирование числа без хвостовых нулей. */
void expr_format(double v, char *buf, size_t n);

#endif /* DC_EXPR_H */
