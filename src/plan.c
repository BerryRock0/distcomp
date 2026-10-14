#include "plan.h"
#include "json.h"
#include "tty.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *state_name(step_state s)
{
    switch (s) {
    case ST_PENDING: return "ожидает";
    case ST_RUNNING: return "выполняется";
    case ST_DONE:    return "готово";
    case ST_FAILED:  return "ошибка";
    case ST_BLOCKED: return "заблокировано";
    }
    return "?";
}

const char *state_icon(step_state s)
{
    switch (s) {
    case ST_PENDING: return "·";
    case ST_RUNNING: return "»";
    case ST_DONE:    return "+";
    case ST_FAILED:  return "x";
    case ST_BLOCKED: return "-";
    }
    return "?";
}

int state_color(step_state s)
{
    switch (s) {
    case ST_DONE:    return C_GREEN;
    case ST_FAILED:  return C_RED;
    case ST_BLOCKED: return C_YELLOW;
    case ST_RUNNING: return C_BLUE;
    case ST_PENDING: return C_DIM;
    }
    return C_RESET;
}

plan *plan_new(const char *goal)
{
    plan *p = xcalloc(1, sizeof *p);
    p->goal = xstrdup(goal ? goal : "");
    return p;
}

void plan_free(plan *p)
{
    if (!p) return;
    for (int i = 0; i < p->n; i++) {
        plan_step *s = &p->steps[i];
        free(s->id);
        free(s->role);
        free(s->title);
        free(s->payload);
        free(s->output);
        free(s->error);
        free(s->ep);
        free(s->pin);
        for (int j = 0; j < s->ndeps; j++) free(s->deps[j]);
        free(s->deps);
    }
    free(p->steps);
    free(p->order);
    free(p->goal);
    free(p->raw);
    free(p);
}

int plan_step_idx(const plan *p, const char *id)
{
    if (!p || !id) return -1;
    for (int i = 0; i < p->n; i++)
        if (strcmp(p->steps[i].id, id) == 0) return i;
    return -1;
}

int plan_add_step(plan *p, const char *id, const char *role, const char *title,
                  const char *payload, const char *const *deps, int ndeps)
{
    plan_step *ns = xrealloc(p->steps, (size_t)(p->n + 1) * sizeof *p->steps);
    p->steps = ns;
    plan_step *s = &p->steps[p->n];
    memset(s, 0, sizeof *s);
    s->id = xstrdup(id);
    s->role = xstrdup(role);
    s->title = xstrdup(title);
    s->payload = xstrdup(payload);
    s->state = ST_PENDING;
    if (ndeps > 0) {
        s->deps = xcalloc((size_t)ndeps, sizeof *s->deps);
        for (int i = 0; i < ndeps; i++) s->deps[i] = xstrdup(deps[i]);
        s->ndeps = ndeps;
    }
    return p->n++;
}

int plan_topo(plan *p, char *err, size_t errlen)
{
    int  *indeg = xcalloc((size_t)p->n, sizeof *indeg);
    int  *order = xcalloc((size_t)p->n, sizeof *order);
    char *done = xcalloc((size_t)p->n, 1);

    for (int i = 0; i < p->n; i++)
        for (int j = 0; j < p->steps[i].ndeps; j++) {
            int d = plan_step_idx(p, p->steps[i].deps[j]);
            if (d < 0) continue;
            indeg[i]++;
        }

    int n = 0;
    for (;;) {
        int pick = -1;
        for (int i = 0; i < p->n; i++) {
            if (done[i] || indeg[i] != 0) continue;
            pick = i;
            break; /* детерминированно: самый ранний индекс */
        }
        if (pick < 0) break;
        done[pick] = 1;
        order[n++] = pick;
        for (int i = 0; i < p->n; i++)
            for (int j = 0; j < p->steps[i].ndeps; j++) {
                int d = plan_step_idx(p, p->steps[i].deps[j]);
                if (d == pick) indeg[i]--;
            }
    }

    free(indeg);
    free(done);
    if (n != p->n) {
        free(order);
        if (err && errlen)
            snprintf(err, errlen, "в графе шагов есть цикл (%d из %d шагов упорядочимы)", n, p->n);
        return -1;
    }
    free(p->order);
    p->order = order;
    p->norder = n;
    return 0;
}

plan *plan_from_json(const char *json, const dc_config *cfg, char *err, size_t errlen)
{
    char     perr[256] = {0};
    js_doc  *doc = js_parse(json, perr, sizeof perr);
    if (!doc) {
        snprintf(err, errlen, "план не является корректным JSON: %s", perr);
        return NULL;
    }
    const js_value *root = js_root(doc);
    if (root->type != JS_OBJ) {
        snprintf(err, errlen, "план должен быть JSON-объектом, а не %s", js_type_name(root->type));
        js_doc_free(doc);
        return NULL;
    }

    const js_value *steps = js_get(root, "steps");
    if (!steps || steps->type != JS_ARR || steps->len == 0) {
        snprintf(err, errlen, "в плане нет массива \"steps\" либо он пуст");
        js_doc_free(doc);
        return NULL;
    }
    if (steps->len > 64) {
        snprintf(err, errlen, "слишком много шагов (%zu), максимум 64", steps->len);
        js_doc_free(doc);
        return NULL;
    }

    const char *goal = js_get_str(root, "goal");
    plan       *p = plan_new(goal ? goal : "");

    /* первый проход: сами шаги */
    for (size_t i = 0; i < steps->len; i++) {
        const js_value *sv = js_at(steps, i);
        if (!sv || sv->type != JS_OBJ) {
            snprintf(err, errlen, "шаг %zu — не объект", i + 1);
            goto fail;
        }
        char idbuf[32];
        const char *id = js_get_str(sv, "id");
        if (!id || !*id) {
            snprintf(idbuf, sizeof idbuf, "s%zu", i + 1);
            id = idbuf;
        }
        if (plan_step_idx(p, id) >= 0) {
            snprintf(err, errlen, "дублирующийся id шага: \"%s\"", id);
            goto fail;
        }
        const char *role = js_get_str(sv, "role");
        if (!role || !*role) role = ROLE_COMPUTE;

        if (!conf_role((dc_config *)cfg, role, 0)) {
            strbuf known;
            sb_init(&known);
            for (int k = 0; k < cfg->nroles; k++) {
                sb_add(&known, known.len ? ", " : "");
                sb_add(&known, cfg->roles[k].name);
            }
            snprintf(err, errlen,
                     "шаг \"%s\": роль \"%s\" не назначена ни одному узлу "
                     "(настроенные роли: %s). Задайте узла в конфигурации "
                     "([role %s] daemon = http://host:port) или через --role %s=URL",
                     id, role, known.p ? known.p : "нет",
                     role, role);
            sb_free(&known);
            goto fail;
        }

        const char *title = js_get_str(sv, "title");
        const char *payload = js_get_str(sv, "payload");
        if (!payload || !*payload) payload = title ? title : id;
        if (!title || !*title) title = payload;

        /* node — необязательное закрепление шага за конкретным узлом */
        const char *pin = js_get_str(sv, "node");
        if (pin && !*pin) pin = NULL;
        if (pin) {
            dc_role *rl = conf_role((dc_config *)cfg, role, 0);
            int      found = 0;
            for (int k = 0; rl && k < rl->nep; k++)
                if (strcmp(rl->eps[k].url, pin) == 0) found = 1;
            if (!found) {
                snprintf(err, errlen,
                         "шаг \"%s\" закреплён за узлом %s, но он не обслуживает "
                         "роль \"%s\"",
                         id, pin, role);
                goto fail;
            }
        }

        /* зависимости собираем во втором проходе */
        const js_value *deps = js_get(sv, "deps");
        const char    *dbuf[64];
        int            nd = 0;
        if (deps && deps->type == JS_ARR) {
            for (size_t k = 0; k < deps->len && nd < 64; k++) {
                const char *d = js_str(js_at(deps, k));
                if (d && *d) dbuf[nd++] = d;
            }
        } else if (deps && deps->type == JS_STR) {
            dbuf[nd++] = deps->str;
        }
        int si = plan_add_step(p, id, role, title, payload, dbuf, nd);
        if (pin) p->steps[si].pin = xstrdup(pin);
    }

    /* второй проход: проверяем зависимости и топологию */
    for (int i = 0; i < p->n; i++)
        for (int j = 0; j < p->steps[i].ndeps; j++) {
            if (plan_step_idx(p, p->steps[i].deps[j]) < 0) {
                snprintf(err, errlen, "шаг \"%s\" зависит от несуществующего \"%s\"",
                         p->steps[i].id, p->steps[i].deps[j]);
                goto fail;
            }
        }

    if (plan_topo(p, err, errlen) != 0) goto fail;

    p->raw = xstrdup(json);
    js_doc_free(doc);
    return p;

fail:
    js_doc_free(doc);
    plan_free(p);
    return NULL;
}

char *plan_to_json(const plan *p)
{
    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\n  \"goal\": ");
    sb_add_json_string(&b, p->goal);
    sb_add(&b, ",\n  \"steps\": [\n");
    for (int i = 0; i < p->n; i++) {
        const plan_step *s = &p->steps[i];
        sb_add(&b, "    { \"id\": ");
        sb_add_json_string(&b, s->id);
        sb_add(&b, ", \"role\": ");
        sb_add_json_string(&b, s->role);
        sb_add(&b, ", \"title\": ");
        sb_add_json_string(&b, s->title);
        sb_add(&b, ", \"payload\": ");
        sb_add_json_string(&b, s->payload);
        if (s->pin) {
            sb_add(&b, ", \"node\": ");
            sb_add_json_string(&b, s->pin);
        }
        sb_add(&b, ", \"deps\": [");
        for (int j = 0; j < s->ndeps; j++) {
            if (j) sb_add(&b, ", ");
            sb_add_json_string(&b, s->deps[j]);
        }
        sb_printf(&b, "] }%s\n", i + 1 < p->n ? "," : "");
    }
    sb_add(&b, "  ]\n}\n");
    return b.p;
}

char *plan_result_json(const plan *p, const char *run_id)
{
    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\n  \"run_id\": ");
    sb_add_json_string(&b, run_id ? run_id : "");
    sb_add(&b, ",\n  \"goal\": ");
    sb_add_json_string(&b, p->goal);
    sb_add(&b, ",\n  \"ok\": ");
    int all_ok = 1;
    for (int i = 0; i < p->n; i++)
        if (p->steps[i].state != ST_DONE) all_ok = 0;
    sb_add(&b, all_ok ? "true" : "false");
    sb_add(&b, ",\n  \"steps\": [\n");
    for (int i = 0; i < p->n; i++) {
        const plan_step *s = &p->steps[i];
        sb_add(&b, "    { \"id\": ");
        sb_add_json_string(&b, s->id);
        sb_add(&b, ", \"role\": ");
        sb_add_json_string(&b, s->role);
        sb_add(&b, ", \"title\": ");
        sb_add_json_string(&b, s->title);
        sb_add(&b, ", \"state\": ");
        sb_add_json_string(&b, state_name(s->state));
        sb_add(&b, ", \"endpoint\": ");
        sb_add_json_string(&b, s->ep ? s->ep : "");
        if (s->pin) {
            sb_add(&b, ",\"node\": ");
            sb_add_json_string(&b, s->pin);
        }
        sb_printf(&b, ", \"ms\": %ld", s->ms);
        sb_printf(&b, ", \"attempts\": %d", s->attempt);
        sb_add(&b, ", \"output\": ");
        sb_add_json_string(&b, s->output ? s->output : "");
        sb_add(&b, ", \"error\": ");
        sb_add_json_string(&b, s->error ? s->error : "");
        sb_printf(&b, " }%s\n", i + 1 < p->n ? "," : "");
    }
    sb_add(&b, "  ]\n}\n");
    return b.p;
}
