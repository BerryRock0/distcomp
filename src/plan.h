#ifndef DC_PLAN_H
#define DC_PLAN_H

#include "conf.h"

#include <stddef.h>

typedef enum {
    ST_PENDING = 0,
    ST_RUNNING,
    ST_DONE,
    ST_FAILED,
    ST_BLOCKED
} step_state;

typedef struct {
    char      *id;
    char      *role;
    char      *title;
    char      *payload;
    char     **deps;
    int        ndeps;
    /* Узел, которому шаг закреплён (адрес узла). Пусто — узел выбирает
     * планировщик по загрузке. Нужен там, где выбор недопустим: слои
     * модели лежат на конкретных устройствах, и отдать слой на другое
     * устройство, где его нет, — значит потерять модель. */
    char      *pin;
    step_state state;
    int        attempt;
    char      *output;
    char      *error;
    char      *ep;        /* адрес узла, выполнившего шаг (владеет шаг) */
    long       ms;
} plan_step;

typedef struct {
    char      *goal;
    char      *raw;       /* исходный JSON плана */
    plan_step *steps;
    int        n;
    int       *order;     /* топологический порядок индексов */
    int        norder;
} plan;

plan *plan_from_json(const char *json, const dc_config *cfg, char *err, size_t errlen);
plan *plan_new(const char *goal);
void  plan_free(plan *p);

int   plan_add_step(plan *p, const char *id, const char *role, const char *title,
                    const char *payload, const char *const *deps, int ndeps);

int   plan_step_idx(const plan *p, const char *id);
int   plan_topo(plan *p, char *err, size_t errlen);      /* заполняет order; -1 при цикле */
char *plan_to_json(const plan *p);
char *plan_result_json(const plan *p, const char *run_id);

const char *state_name(step_state s);
const char *state_icon(step_state s);

/* Цвет состояния для терминала (tty.h). Панель и CLI показывают одно и то же. */
int state_color(step_state s);

#endif /* DC_PLAN_H */
