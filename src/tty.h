#ifndef DC_TTY_H
#define DC_TTY_H

#include <stdio.h>

/*
 * Цвет в терминале. Включается сам, если stdout — терминал и в окружении
 * нет NO_COLOR; выключается флагом --color. Когда вывод уходит в файл или
 * в конвейер (лог, тест), коды не подмешиваются и текст остаётся пригодным
 * для grep.
 */
typedef enum { C_RESET = 0, C_DIM, C_BOLD, C_GREEN, C_RED, C_YELLOW, C_BLUE, C_CYAN } dc_color;

void tty_init(int force_mode);   /* -1 авто, 0 нет, 1 да */
int  tty_color(void);
void tty_set(FILE *f, dc_color c);

/* Печать с цветом в одну строку: printf-строка с одним %s на цвет. */
#define TC(f, color, fmt, ...)                     \
    do {                                           \
        tty_set((f), (color));                     \
        fprintf((f), (fmt), ##__VA_ARGS__);       \
        tty_set((f), C_RESET);                     \
    } while (0)

#endif /* DC_TTY_H */