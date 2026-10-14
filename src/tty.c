#include "tty.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * Решение принимается один раз, при разборе аргументов: терминал может
 * перенаправить вывод уже после этого, но логика «цвет или нет» должна быть
 * одинаковой для всех строк запуска, иначе отчёт соберётся из разного.
 */
static int mode = -1;   /* -1 ещё не решали, 0 нет, 1 да */

void tty_init(int force_mode)
{
    if (force_mode >= 0) {
        mode = force_mode;
        return;
    }
    const char *term = getenv("TERM");
    mode = isatty(STDOUT_FILENO) && !getenv("NO_COLOR") && term && strcmp(term, "dumb") != 0;
}

int tty_color(void)
{
    if (mode < 0) tty_init(-1);
    return mode;
}

void tty_set(FILE *f, dc_color c)
{
    if (!tty_color()) return;
    const char *code = "\033[0m";
    switch (c) {
    case C_DIM:    code = "\033[2m";  break;
    case C_BOLD:   code = "\033[1m";  break;
    case C_GREEN:  code = "\033[32m"; break;
    case C_RED:    code = "\033[31m"; break;
    case C_YELLOW: code = "\033[33m"; break;
    case C_BLUE:   code = "\033[34m"; break;
    case C_CYAN:   code = "\033[36m"; break;
    case C_RESET:  code = "\033[0m";  break;
    }
    fputs(code, f);
}