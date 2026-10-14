#include "web.h"
#include "json.h"
#include "llm.h"
#include "plan.h"
#include "sched.h"
#include "util.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define MAX_BODY   (1 << 20)
#define MAX_RUNS   64

/* ------------------------------------------------------- реестр ------ */

typedef struct run_rec {
    struct dc_web   *w;
    char            id[32];
    char            goal[2048];
    char   phase[16];   /* planning | running | done */
    int             execute;    /* 0 — только построить план */
    /* Готовый план, присланный из панели после правки человеком: тогда LLM
     * не зовётся вовсе, а на узлы уходит ровно то, что человек утвердил. */
    char           *plan_json;
    plan           *p;           /* план принадлежит записи */
    dc_config      *cfg;         /* снимок конфигурации на момент запуска */
    int             finished;
    int             ok;
    long            started_ms;
    long            ended_ms;
    char            error[512];
    struct run_rec *next;
} run_rec;

#define PROBE_TTL_MS 3000   /* как долго счётчики и состояния узлов считаются свежими */

struct dc_web {
    dc_config       *cfg;
    pthread_mutex_t  mu;         /* защищает roles во время правок из UI */
    pthread_mutex_t  run_mu;     /* защищает список запусков */
    run_rec         *runs;
    /* последний опрос узлов: панель спрашивает состояние раз в 1.2 с, а узлы
     * отвечают на PING по сети — без кэша это до восьми лишних обменов в
     * секунду на каждый узел. Данные успевают устареть на 3 с, что для панели
     * незаметно, а сеть перестаёт зря просыпаться. */
    dc_node_status  probe[DC_MAX_ROLES * DC_MAX_EPS];
    int             probe_n;
    long            probe_at;
    int              listen_fd;
    int              port;
    char             api_base[32];
    int              no_llm;
    volatile int     stopping;
};

/* -------------------------------------------------------- утилиты ----- */

static void url_decode(const char *in, char *out, size_t outsz)
{
    size_t o = 0;
    for (size_t i = 0; in[i] && o + 1 < outsz; i++) {
        if (in[i] == '+') {
            out[o++] = ' ';
        } else if (in[i] == '%' && in[i + 1] && in[i + 2]) {
            char hex[3] = { in[i + 1], in[i + 2], 0 };
            out[o++] = (char)strtol(hex, NULL, 16);
            i += 2;
        } else {
            out[o++] = in[i];
        }
    }
    out[o] = '\0';
}

static const char *query_param(const char *query, const char *name, char *buf, size_t bufsz)
{
    size_t nl = strlen(name);
    while (query && *query) {
        const char *amp = strchr(query, '&');
        size_t      seglen = amp ? (size_t)(amp - query) : strlen(query);
        if (seglen > nl && strncmp(query, name, nl) == 0 && query[nl] == '=') {
            size_t vlen = seglen - nl - 1;
            if (vlen >= bufsz) vlen = bufsz - 1;
            char tmp[1024];
            if (vlen >= sizeof tmp) vlen = sizeof tmp - 1;
            memcpy(tmp, query + nl + 1, vlen);
            tmp[vlen] = '\0';
            url_decode(tmp, buf, bufsz);
            return buf;
        }
        if (!amp) break;
        query = amp + 1;
    }
    return NULL;
}

static void http_reply(int fd, int status, const char *ctype, const char *body, size_t len)
{
    const char *reason = status == 200 ? "OK"
                       : status == 404 ? "Not Found"
                       : status == 400 ? "Bad Request"
                       : status == 409 ? "Conflict"
                       : status == 500 ? "Internal Server Error"
                                       : "Error";
    char head[512];
    int  hn = snprintf(head, sizeof head,
                       "HTTP/1.1 %d %s\r\n"
                       "Content-Type: %s\r\n"
                       "Content-Length: %zu\r\n"
                       "Cache-Control: no-store\r\n"
                       "Connection: close\r\n\r\n",
                       status, reason, ctype, len);
    /* snprintf вернул бы «длину без обрезки»: без клампа write читал бы
     * за пределами стекового буфера */
    if (hn < 0 || (size_t)hn >= sizeof head) hn = (int)sizeof head - 1;
    if (write(fd, head, (size_t)hn) < 0) return;
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, body + off, len - off);
        if (w <= 0) break;
        off += (size_t)w;
    }
}

static void reply_json(int fd, int status, const strbuf *b)
{
    http_reply(fd, status, "application/json; charset=utf-8", b->p ? b->p : "{}",
               b->p ? b->len : 2);
}

static void reply_error(int fd, int status, const char *msg)
{
    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\"error\":");
    sb_add_json_string(&b, msg);
    sb_add(&b, "}");
    reply_json(fd, status, &b);
    sb_free(&b);
}

/* Панель спрашивает то же, что и планировщик, иначе они разойдутся во мнении
 * о готовности LLM. */
/* Опрос идёт на копии конфигурации, иначе правка ролей из другой вкладки
 * гонялась бы с записью в память. Но выясненные лимиты и память узлов — это
 * знание, а не состояние: переносим их обратно, чтобы панель показывала
 * реальные числа, а нули, и чтобы планировщик не опрашивал узлы повторно. */
static void caps_copy_back(dc_config *dst, const dc_config *src)
{
    for (int i = 0; i < src->nroles; i++) {
        dc_role *d = conf_role(dst, src->roles[i].name, 0);
        if (!d) continue;
        for (int j = 0; j < src->roles[i].nep; j++) {
            for (int k = 0; k < d->nep; k++) {
                if (strcmp(d->eps[k].url, src->roles[i].eps[j].url) != 0) continue;
                d->eps[k].cap = src->roles[i].eps[j].cap;
                d->eps[k].completed = src->roles[i].eps[j].completed;
                d->eps[k].failed = src->roles[i].eps[j].failed;
            }
        }
    }
}

static int llm_available_for_web(const dc_config *cfg)
{
    if (!cfg->llm_url || !*cfg->llm_url || !cfg->llm_model || !*cfg->llm_model) return 0;
    return !llm_needs_key(cfg);
}

/* ------------------------------------------------------- состояние ---- */

static void json_plan(strbuf *b, const plan *p)
{
    sb_add(b, "\"steps\":[");
    for (int i = 0; i < p->n; i++) {
        const plan_step *s = &p->steps[i];
        if (i) sb_add(b, ",");
        sb_add(b, "{\"id\":");
        sb_add_json_string(b, s->id);
        sb_add(b, ",\"role\":");
        sb_add_json_string(b, s->role);
        sb_add(b, ",\"title\":");
        sb_add_json_string(b, s->title);
        sb_add(b, ",\"payload\":");
        sb_add_json_string(b, s->payload);
        sb_add(b, ",\"deps\":[");
        for (int k = 0; k < s->ndeps; k++) {
            if (k) sb_add(b, ",");
            sb_add_json_string(b, s->deps[k]);
        }
        if (s->pin) {
            sb_add(b, ",\"node\":");
            sb_add_json_string(b, s->pin);
        }
        sb_add(b, "],\"state\":");
        sb_add_json_string(b, state_name(s->state));
        sb_add(b, ",\"endpoint\":");
        sb_add_json_string(b, s->ep ? s->ep : "");
        sb_printf(b, ",\"ms\":%ld,\"attempts\":%d", s->ms, s->attempt);
        sb_add(b, ",\"output\":");
        sb_add_json_string(b, s->output ? s->output : "");
        sb_add(b, ",\"error\":");
        sb_add_json_string(b, s->error ? s->error : "");
        sb_add(b, "}");
    }
    sb_add(b, "]");
}

/* close=1 — самостоятельный объект (список запусков);
 * close=0 — открытый объект без «}», чтобы дописать «,"plan":…}». */
static void json_run_summary(strbuf *b, const run_rec *r, int close)
{
    sb_add(b, "{\"id\":");
    sb_add_json_string(b, r->id);
    sb_add(b, ",\"goal\":");
    sb_add_json_string(b, r->goal);
    sb_add(b, ",\"phase\":");
    sb_add_json_string(b, r->phase);
    sb_printf(b, ",\"finished\":%s", r->finished ? "true" : "false");
    sb_printf(b, ",\"ok\":%s", r->ok ? "true" : "false");
    sb_printf(b, ",\"started_ms\":%ld", r->started_ms);
    sb_printf(b, ",\"ended_ms\":%ld", r->ended_ms);
    sb_add(b, ",\"error\":");
    sb_add_json_string(b, r->error);
    if (close) sb_addc(b, '}');
}

/* ------------------------------------------------------ выполнение ---- */

static run_rec *run_find(dc_web *w, const char *id)
{
    for (run_rec *r = w->runs; r; r = r->next)
        if (strcmp(r->id, id) == 0) return r;
    return NULL;
}

static void run_finish(run_rec *r, int ok, const char *err)
{
    pthread_mutex_lock(&r->w->run_mu);
    snprintf(r->phase, sizeof r->phase, "done");
    r->finished = 1;
    r->ok = ok;
    r->ended_ms = now_ms();
    if (err && *err) snprintf(r->error, sizeof r->error, "%s", err);
    pthread_mutex_unlock(&r->w->run_mu);
}

/* Поток выполнения НЕ удаляет запись: она лежит в реестре w->runs и нужна
 * для истории и для панели, которая опрашивает статус уже после конца запуска.
 * Освобождается всё только в web_free(). Здесь — лишь снимок конфигурации. */
static void run_release(run_rec *r)
{
    conf_free(r->cfg);
    r->cfg = NULL;
    free(r->plan_json);
    r->plan_json = NULL;
}

static void *run_thread(void *arg)
{
    run_rec *r = arg;

    char  err[1024] = {0};
    char *plan_json = NULL;

    if (r->plan_json) {
        /* план утвердил человек: LLM не участвует, узлы получат ровно то,
         * что человек отредактировал в панели */
        log_info("план прислан из панели, LLM не вызывается");
        plan_json = xstrdup(r->plan_json);
    } else {
        /* возможности узлов нужны до промпта: модель должна знать лимит тела,
         * иначе придумает шаг, который микроконтроллер не примет */
        if (!r->w->no_llm && r->cfg) sched_learn_caps(r->cfg);
        plan_json = llm_make_plan(r->cfg, r->goal, !r->w->no_llm, 1, err, sizeof err);
    }

    if (!plan_json) {
        run_finish(r, 0, err);
        run_release(r);
        return NULL;
    }

    r->p = plan_from_json(plan_json, r->cfg, err, sizeof err);
    free(plan_json);
    if (!r->p) {
        char msg[1200];
        snprintf(msg, sizeof msg, "план отклонён: %s", err);
        run_finish(r, 0, msg);
        run_release(r);
        return NULL;
    }

    /* сохраняем план на диск, чтобы его можно было переиграть из CLI */
    char dir[1024];
    if (r->cfg->outdir && *r->cfg->outdir) {
        snprintf(dir, sizeof dir, "%s/%s", r->cfg->outdir, r->id);
        for (char *p = dir + 1; *p; p++) {
            if (*p == '/') {
                *p = '\0';
                mkdir(dir, 0775);
                *p = '/';
            }
        }
        mkdir(dir, 0775);
        char path[1200], *pretty;
        snprintf(path, sizeof path, "%s/plan.json", dir);
        if ((pretty = plan_to_json(r->p))) {
            FILE *f = fopen(path, "w");
            if (f) {
                fputs(pretty, f);
                fclose(f);
            }
            free(pretty);
        }
    }

    if (!r->execute) {
        /* режим просмотра: план построен, но ничего не отправляем узлам */
        run_finish(r, 1, NULL);
        run_release(r);
        return NULL;
    }

    pthread_mutex_lock(&r->w->run_mu);
    snprintf(r->phase, sizeof r->phase, "running");
    pthread_mutex_unlock(&r->w->run_mu);

    sched_run(r->p, r->cfg);

    if (r->cfg->outdir && *r->cfg->outdir) {
        char path[1200], *res;
        snprintf(path, sizeof path, "%s/result.json", dir);
        res = plan_result_json(r->p, r->id);
        FILE *f = fopen(path, "w");
        if (f) {
            fputs(res, f);
            fclose(f);
        }
        free(res);
    }

    int ok = 1;
    for (int i = 0; i < r->p->n; i++)
        if (r->p->steps[i].state != ST_DONE) ok = 0;

    run_finish(r, ok, NULL);
    run_release(r);
    return NULL;
}

/* --------------------------------------------------------- маршруты --- */

static void api_state(dc_web *w, int fd)
{
    strbuf b;
    sb_init(&b);

    /* Опрос узлов идёт по сети — по секунде на каждый. Спим блокировку на это
     * время, иначе панель подвисала бы целиком (до обновления и до правки
     * ролей) из-за одного неотвечающего узла в другой комнате. */
    pthread_mutex_lock(&w->mu);
    int need_probe = now_ms() - w->probe_at > PROBE_TTL_MS;
    pthread_mutex_unlock(&w->mu);

    dc_node_status ns[DC_MAX_ROLES * DC_MAX_EPS];
    int            n = 0;
    if (need_probe) {
        pthread_mutex_lock(&w->mu);
        dc_config *probe_cfg = conf_clone(w->cfg);
        pthread_mutex_unlock(&w->mu);

        n = probe_cfg ? sched_probe(probe_cfg, ns, (int)(sizeof ns / sizeof ns[0])) : 0;

        pthread_mutex_lock(&w->mu);
        if (probe_cfg) caps_copy_back(w->cfg, probe_cfg);
        memcpy(w->probe, ns, sizeof ns);
        w->probe_n = n;
        w->probe_at = now_ms();
        pthread_mutex_unlock(&w->mu);
        conf_free(probe_cfg);
    } else {
        pthread_mutex_lock(&w->mu);
        memcpy(ns, w->probe, sizeof ns);
        n = w->probe_n;
        pthread_mutex_unlock(&w->mu);
    }

    pthread_mutex_lock(&w->mu);
    sb_add(&b, "{\"roles\":[");
    for (int i = 0; i < w->cfg->nroles; i++) {
        dc_role *r = &w->cfg->roles[i];
        if (i) sb_add(&b, ",");
        sb_add(&b, "{\"name\":");
        sb_add_json_string(&b, r->name);
        sb_add(&b, ",\"nodes\":[");
        for (int j = 0; j < r->nep; j++) {
            if (j) sb_add(&b, ",");
            sb_add(&b, "{\"url\":");
            sb_add_json_string(&b, r->eps[j].url);
            dc_capability *cap = &r->eps[j].cap;
            if (cap->known) {
                sb_add(&b, ",\"max_body\":");
                sb_printf(&b, "%d", cap->max_body);
                sb_add(&b, ",\"arch\":");
                sb_add_json_string(&b, cap->arch);
            } else {
                sb_add(&b, ",\"max_body\":0,\"arch\":\"\"");
            }
            sb_printf(&b, ",\"completed\":%ld,\"failed\":%ld", r->eps[j].completed,
                      r->eps[j].failed);
            sb_add(&b, "}");
        }
        sb_add(&b, "]}");
    }
    sb_add(&b, "],\"nodes\":[");
    for (int i = 0; i < n; i++) {
        if (i) sb_add(&b, ",");
        sb_add(&b, "{\"role\":");
        sb_add_json_string(&b, ns[i].role);
        sb_add(&b, ",\"url\":");
        sb_add_json_string(&b, ns[i].url);
        sb_add(&b, ",\"host\":");
        sb_add_json_string(&b, ns[i].host);
        sb_printf(&b, ",\"alive\":%s", ns[i].alive ? "true" : "false");
        sb_printf(&b, ",\"ms\":%d", ns[i].ms);
        sb_add(&b, ",\"max_body\":");
        sb_printf(&b, "%d", ns[i].max_body);
        /* сколько памяти узел отдаёт под веса модели */
        sb_printf(&b, ",\"free_mem\":%ld", ns[i].free_mem);
        sb_add(&b, ",\"shard\":");
        sb_add_json_string(&b, ns[i].shard);
        sb_add(&b, ",\"arch\":");
        sb_add_json_string(&b, ns[i].arch);
        sb_add(&b, ",\"detail\":");
        sb_add_json_string(&b, ns[i].detail);
        sb_add(&b, "}");
    }
    sb_add(&b, "],\"settings\":{");
    sb_printf(&b, "\"llm_model\":");
    sb_add_json_string(&b, w->cfg->llm_model);
    /* «настроена» = к LLM можно обратиться: ключ есть, либо сервер локальный
     * и ключ ему не нужен. Иначе панель врёт про локальный сервер без ключа */
    sb_printf(&b, ",\"llm_configured\":%s", llm_available_for_web(w->cfg) ? "true" : "false");
    sb_add(&b, ",\"llm_url\":");
    sb_add_json_string(&b, w->cfg->llm_url);
    sb_add(&b, ",\"llm_needs_key\":");
    sb_add(&b, llm_needs_key(w->cfg) ? "true" : "false");
    sb_printf(&b, ",\"payload_mode\":");
    sb_add_json_string(&b, w->cfg->payload_mode);
    sb_printf(&b, ",\"concurrency\":%d,\"retries\":%d,\"timeout_ms\":%ld",
              w->cfg->concurrency, w->cfg->retries, w->cfg->timeout_ms);
    sb_add(&b, "}");   /* закрываем settings; внешний объект закроется в конце */

    /* Список запусков в панели не растёт бесконечно: MAX_RUNS последних.
     * Старое не пропадает из runs/ на диске — там оно и живёт как история. */
    pthread_mutex_lock(&w->run_mu);
    sb_add(&b, ",\"runs\":[");
    int first = 1, shown = 0;
    for (run_rec *r = w->runs; r && shown < MAX_RUNS; r = r->next, shown++) {
        if (!first) sb_add(&b, ",");
        first = 0;
        json_run_summary(&b, r, 1);
    }
    sb_add(&b, "]}");
    pthread_mutex_unlock(&w->run_mu);

    pthread_mutex_unlock(&w->mu);
    reply_json(fd, 200, &b);
    sb_free(&b);
}

static void api_run_detail(dc_web *w, int fd, const char *id)
{
    pthread_mutex_lock(&w->run_mu);
    run_rec *r = run_find(w, id);
    if (!r) {
        pthread_mutex_unlock(&w->run_mu);
        reply_error(fd, 404, "запуск не найден");
        return;
    }
    strbuf b;
    sb_init(&b);
    json_run_summary(&b, r, 0);
    sb_add(&b, ",\"plan\":");
    if (r->p) {
        sb_add(&b, "{\"goal\":");
        sb_add_json_string(&b, r->p->goal);
        sb_add(&b, ",");
        json_plan(&b, r->p);
        sb_add(&b, "}");
    } else {
        sb_add(&b, "null");
    }
    /* план без результатов: его показывают в редакторе перед запуском */
    sb_add(&b, ",\"plan_json\":");
    char *pretty = r->p ? plan_to_json(r->p) : NULL;
    if (pretty) {
        sb_add_json_string(&b, pretty);
        free(pretty);
    } else {
        sb_add(&b, "null");
    }
    sb_add(&b, "}");
    pthread_mutex_unlock(&w->run_mu);
    reply_json(fd, 200, &b);
    sb_free(&b);
}

static void api_start_run(dc_web *w, int fd, const char *body, int execute)
{
    char    perr[256] = {0};
    js_doc *doc = js_parse(body && *body ? body : "{}", perr, sizeof perr);
    if (!doc) {
        reply_error(fd, 400, perr);
        return;
    }
    const char *goal = js_get_str(js_root(doc), "goal");
    const char *plan_json = js_get_str(js_root(doc), "plan");
    if ((!goal || !*goal) && (!plan_json || !*plan_json)) {
        js_doc_free(doc);
        reply_error(fd, 400, "не задана цель");
        return;
    }

    run_rec *r = xcalloc(1, sizeof *r);
    r->w = w;
    r->execute = execute;
    r->plan_json = plan_json && *plan_json ? xstrdup(plan_json) : NULL;
    if (goal && *goal) snprintf(r->goal, sizeof r->goal, "%s", goal);
    else              snprintf(r->goal, sizeof r->goal, "план прислан из панели");
    /* снимок конфигурации под w->mu: правка роли из другого потока могла
     * бы освободить url прямо во время копирования */
    pthread_mutex_lock(&w->mu);
    r->cfg = conf_clone(w->cfg);
    pthread_mutex_unlock(&w->mu);

    pthread_mutex_lock(&w->run_mu);
    if (w->runs && !w->runs->finished) {
        pthread_mutex_unlock(&w->run_mu);
        conf_free(r->cfg);
        free(r->plan_json);
        free(r);
        js_doc_free(doc);
        reply_error(fd, 409, "предыдущий запуск ещё идёт — дождитесь его или откройте новую вкладку");
        return;
    }
    /* метка из времени и счётчика: несколько запусков в одну секунду не сольются */
    time_t    t = time(NULL);
    struct tm tmv;
    localtime_r(&t, &tmv);
    static int seq = 0;
    snprintf(r->id, sizeof r->id, "%04d%02d%02d-%02d%02d%02d-%02d",
             (tmv.tm_year + 1900) % 10000, (tmv.tm_mon + 1) % 100, tmv.tm_mday % 100,
             tmv.tm_hour % 100, tmv.tm_min % 100, tmv.tm_sec % 100, ++seq % 100);
    snprintf(r->goal, sizeof r->goal, "%s", goal);
    snprintf(r->phase, sizeof r->phase, "planning");
    r->started_ms = now_ms();
    r->next = w->runs;
    w->runs = r;
    pthread_mutex_unlock(&w->run_mu);
    js_doc_free(doc);

    pthread_t th;
    if (pthread_create(&th, NULL, run_thread, r) != 0) {
        run_finish(r, 0, "не удалось создать поток запуска");
        run_release(r);
        reply_error(fd, 500, "не удалось создать поток запуска");
        return;
    }
    pthread_detach(th);

    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\"id\":");
    sb_add_json_string(&b, r->id);
    sb_add(&b, "}");
    reply_json(fd, 200, &b);
    sb_free(&b);
}

static void api_role(dc_web *w, int fd, const char *body)
{
    char    perr[256] = {0};
    js_doc *doc = js_parse(body && *body ? body : "{}", perr, sizeof perr);
    if (!doc) {
        reply_error(fd, 400, perr);
        return;
    }
    const js_value *root = js_root(doc);
    const char     *role = js_get_str(root, "role");
    const char     *url = js_get_str(root, "url");
    const char     *op = js_get_str(root, "op");
    if (!role || !url || !*url) {
        js_doc_free(doc);
        reply_error(fd, 400, "нужны поля role и url");
        return;
    }
    int remove = op && str_ieq(op, "remove");

    pthread_mutex_lock(&w->mu);
    dc_role *r = conf_role(w->cfg, role, !remove);
    if (!r) {
        pthread_mutex_unlock(&w->mu);
        js_doc_free(doc);
        reply_error(fd, 400, "неизвестная роль или лимит ролей исчерпан");
        return;
    }
    if (remove) {
        int found = -1;
        for (int i = 0; i < r->nep; i++)
            if (strcmp(r->eps[i].url, url) == 0) { found = i; break; }
        if (found < 0) {
            pthread_mutex_unlock(&w->mu);
            js_doc_free(doc);
            reply_error(fd, 404, "такого узла в роли нет");
            return;
        }
        free(r->eps[found].url);
        for (int i = found; i + 1 < r->nep; i++) r->eps[i] = r->eps[i + 1];
        r->nep--;
    } else if (conf_role_add_ep(r, url) != 0) {
        pthread_mutex_unlock(&w->mu);
        js_doc_free(doc);
        reply_error(fd, 400, "лимит узлов в роли исчерпан");
        return;
    }
    /* применяем к следующему запуску */
    dc_config *live = conf_clone(w->cfg);
    pthread_mutex_unlock(&w->mu);
    js_doc_free(doc);

    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\"ok\":true,\"roles\":");
    int first = 1;
    for (int i = 0; i < live->nroles; i++) {
        if (!first) sb_add(&b, ",");
        first = 0;
        sb_add(&b, "{\"name\":");
        sb_add_json_string(&b, live->roles[i].name);
        sb_add(&b, ",\"nodes\":[");
        for (int j = 0; j < live->roles[i].nep; j++) {
            if (j) sb_add(&b, ",");
            sb_add_json_string(&b, live->roles[i].eps[j].url);
        }
        sb_add(&b, "]}");
    }
    sb_add(&b, "}");
    reply_json(fd, 200, &b);
    sb_free(&b);
    conf_free(live);
}

/* Проверка LLM по кнопке в панели: отвечает тот же llm_check, что и CLI,
 * но его вывод идёт в ответ, а не в терминал панели. */
static void api_llm(dc_web *w, int fd)
{
    dc_config *cfg = NULL;
    pthread_mutex_lock(&w->mu);
    cfg = conf_clone(w->cfg);
    pthread_mutex_unlock(&w->mu);

    strbuf b;
    sb_init(&b);
    char err[512] = {0};
    int  rc = cfg ? llm_check_quiet(cfg, err, sizeof err) : 3;

    sb_add(&b, "{\"ok\":");
    sb_add(&b, rc == 0 ? "true" : "false");
    sb_add(&b, ",\"error\":");
    sb_add_json_string(&b, rc == 0 ? "" : err);
    sb_add(&b, ",\"url\":");
    sb_add_json_string(&b, cfg && cfg->llm_url ? cfg->llm_url : "");
    sb_add(&b, ",\"model\":");
    sb_add_json_string(&b, cfg && cfg->llm_model ? cfg->llm_model : "");
    sb_add(&b, ",\"models\":[");
    if (rc == 0 && cfg) {
        char **ids = NULL;
        int    n = llm_list_models(cfg, &ids, err, sizeof err);
        for (int i = 0; i < n; i++) {
            if (i) sb_add(&b, ",");
            sb_add_json_string(&b, ids[i]);
            free(ids[i]);
        }
        free(ids);
    }
    sb_add(&b, "]}");
    conf_free(cfg);
    reply_json(fd, rc == 0 ? 200 : 503, &b);
    sb_free(&b);
}

/* Потоки соединений живут отдельно и после web_stop могут ещё дочитывать
 * сокет. Освобождать dc_web под ними нельзя, поэтому после остановки сервера
 * структуру намеренно не трогаем: процесс всё равно завершается, а use-after-free
 * на выходе — это падение вместо тихого выхода. */

/* ------------------------------------------------------- соединение --- */

typedef struct {
    dc_web *w;
    int     fd;
} conn_arg;

static int read_line(int fd, char *buf, size_t n)
{
    size_t i = 0;
    while (i + 1 < n) {
        char c;
        ssize_t r = read(fd, &c, 1);
        if (r <= 0) return -1;
        if (c == '\n') {
            buf[i] = '\0';
            while (i > 0 && buf[i - 1] == '\r') buf[--i] = '\0';
            return (int)i;
        }
        buf[i++] = c;
    }
    buf[i] = '\0';
    return (int)i;
}

/* Соединение живёт недолго: клиент может открыть сокет и замолчать, а поток
 * при этом висит. Один поток на соединение — без этого панель很容易 упрётся
 * в лимит потоков из-за одного зависшего вкладки браузера. */
#define CONN_TIMEOUT_S 30

static void *conn_thread(void *arg)
{
    conn_arg *ca = arg;
    dc_web   *w = ca->w;
    int       fd = ca->fd;

    struct timeval tv = { CONN_TIMEOUT_S, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);

    char reqline[4096];
    if (read_line(fd, reqline, sizeof reqline) < 0) goto done;

    char method[16] = {0}, target[2048] = {0};
    sscanf(reqline, "%15s %2047s", method, target);

    long content_length = 0;
    for (;;) {
        char line[2048];
        if (read_line(fd, line, sizeof line) < 0) goto done;
        if (!line[0]) break;
        if (strncasecmp(line, "Content-Length:", 15) == 0)
            content_length = strtol(line + 15, NULL, 10);
    }

    char *body = NULL;
    if (content_length > 0) {
        if (content_length > MAX_BODY) {
            reply_error(fd, 400, "слишком большой запрос");
            goto done;
        }
        body = xmalloc((size_t)content_length + 1);
        size_t got = 0;
        while (got < (size_t)content_length) {
            ssize_t r = read(fd, body + got, (size_t)content_length - got);
            if (r <= 0) break;
            got += (size_t)r;
        }
        body[got] = '\0';
    }

    char path[2048];
    char query[2048] = {0};
    char *qm = strchr(target, '?');
    if (qm) {
        snprintf(path, sizeof path, "%.*s", (int)(qm - target), target);
        snprintf(query, sizeof query, "%s", qm + 1);
    } else {
        snprintf(path, sizeof path, "%s", target);
    }

    /* маршруты строим от api_base, чтобы его можно было переопределить */
    char base[64], r_state[80], r_run[80], r_role[80], r_llm[80];
    snprintf(base, sizeof base, "%s", w->api_base);
    snprintf(r_state, sizeof r_state, "%s/state", base);
    snprintf(r_run, sizeof r_run, "%s/run", base);
    snprintf(r_role, sizeof r_role, "%s/role", base);
    snprintf(r_llm, sizeof r_llm, "%s/llm", base);

    if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
        http_reply(fd, 200, "text/html; charset=utf-8", web_ui_html(),
                   strlen(web_ui_html()));
    } else if (strcmp(method, "GET") == 0 && strcmp(path, r_state) == 0) {
        api_state(w, fd);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, r_run) == 0) {
        char id[64] = {0};
        if (query_param(query, "id", id, sizeof id)) api_run_detail(w, fd, id);
        else reply_error(fd, 400, "нужен параметр id");
    } else if (strcmp(method, "POST") == 0 && strcmp(path, r_run) == 0) {
        int execute = 1;
        js_doc *d = js_parse(body ? body : "{}", NULL, 0);
        if (d) {
            const js_value *ex = js_get(js_root(d), "execute");
            if (ex && ex->type == JS_BOOL) execute = ex->bval;
            js_doc_free(d);
        }
        api_start_run(w, fd, body, execute);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, r_role) == 0) {
        api_role(w, fd, body);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, r_llm) == 0) {
        api_llm(w, fd);
    } else if (strcmp(method, "GET") == 0 && (strcmp(path, "/favicon.ico") == 0)) {
        /* без этого каждая вкладка оставляет в логе 404 */
        http_reply(fd, 204, "image/x-icon", "", 0);
    } else {
        reply_error(fd, 404, "нет такого маршрута");
    }

    free(body);
done:
    shutdown(fd, SHUT_RDWR);
    close(fd);
    free(ca);
    return NULL;
}

/* ------------------------------------------------------------ main ---- */

dc_web *web_start(dc_config *cfg, const char *host, int port, const char *api_base, int no_llm,
                  char *err, size_t errlen)
{
    dc_web *w = xcalloc(1, sizeof *w);
    w->cfg = cfg;
    w->port = port;
    w->no_llm = no_llm;
    snprintf(w->api_base, sizeof w->api_base, "%s", api_base && *api_base ? api_base : "/api");
    pthread_mutex_init(&w->mu, NULL);
    pthread_mutex_init(&w->run_mu, NULL);

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        snprintf(err, errlen, "socket: %s", strerror(errno));
        web_free(w);
        return NULL;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host ? host : "0.0.0.0", &addr.sin_addr) != 1) {
        snprintf(err, errlen, "неверный адрес: %s", host);
        close(fd);
        web_free(w);
        return NULL;
    }
    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) != 0) {
        snprintf(err, errlen, "не удалось занять %s:%d — %s", host, port, strerror(errno));
        close(fd);
        web_free(w);
        return NULL;
    }
    if (listen(fd, 64) != 0) {
        snprintf(err, errlen, "listen: %s", strerror(errno));
        close(fd);
        web_free(w);
        return NULL;
    }
    w->listen_fd = fd;
    return w;
}

int web_serve(dc_web *w)
{
    while (!w->stopping) {
        int c = accept(w->listen_fd, NULL, NULL);
        if (c < 0) {
            if (errno == EINTR) continue;
            if (w->stopping) break;
            log_error("accept: %s", strerror(errno));
            break;
        }
        conn_arg *ca = xmalloc(sizeof *ca);
        ca->w = w;
        ca->fd = c;
        pthread_t th;
        if (pthread_create(&th, NULL, conn_thread, ca) != 0) {
            close(c);
            free(ca);
            continue;
        }
        pthread_detach(th);
    }
    return 0;
}

void web_stop(dc_web *w)
{
    if (!w) return;
    w->stopping = 1;
    /* accept() вернёт ошибку и цикл web_serve() завершится */
    if (w->listen_fd >= 0) shutdown(w->listen_fd, SHUT_RDWR);
}

int web_port(const dc_web *w) { return w ? w->port : 0; }

void web_free(dc_web *w)
{
    if (!w) return;
    if (w->listen_fd >= 0) close(w->listen_fd);
    run_rec *r = w->runs;
    while (r) {
        run_rec *next = r->next;
        plan_free(r->p);
        conf_free(r->cfg);
        free(r->plan_json);
        free(r);
        r = next;
    }
    pthread_mutex_destroy(&w->mu);
    pthread_mutex_destroy(&w->run_mu);
    free(w);
}
