#include "sched.h"
#include "json.h"
#include "tty.h"
#include "udp.h"
#include "xport.h"
#include "util.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define MAX_SLOTS 64

typedef struct {
    dc_config *cfg;
    plan      *p;
    int        step;
    int        attempt;
    long       started;

    dc_endpoint *ep;
    long         ms;
    int          ok;
    char        *output;
    char        *error;
} job;

typedef struct {
    pthread_mutex_t mu;
    dc_config      *cfg;
    plan           *p;
    job             jobs[MAX_SLOTS];
    pthread_t       th[MAX_SLOTS];
    int             th_step[MAX_SLOTS];
    long            rr;
} runner;

static void lock(runner *r)   { pthread_mutex_lock(&r->mu); }
static void unlock(runner *r) { pthread_mutex_unlock(&r->mu); }

/* Определена ниже, вместе с остальной проверкой узлов. */
static int node_probe(dc_config *cfg, dc_endpoint *ep, int *ms_out, dc_capability *cap,
                       char *err, size_t errlen);

/* Опрос всех узлов сразу, параллельно: последовательный обход добавил бы
 * N·RTT к началу каждого запуска, что заметно на медленной сети. */
typedef struct {
    dc_config *cfg;
    int        role_idx;
    int        ep_idx;
} probe_arg;

static void *probe_one(void *arg)
{
    probe_arg   *a = arg;
    dc_role     *r = &a->cfg->roles[a->role_idx];
    dc_endpoint *e = &r->eps[a->ep_idx];

    char err[512] = {0};
    int  ms = 0;
    node_probe(a->cfg, e, &ms, &e->cap, err, sizeof err);
    return NULL;
}

static void probe_all(dc_config *cfg)
{
    probe_arg args[DC_MAX_ROLES * DC_MAX_EPS];
    pthread_t th[DC_MAX_ROLES * DC_MAX_EPS];
    int       n = 0;

    for (int i = 0; i < cfg->nroles; i++)
        for (int j = 0; j < cfg->roles[i].nep; j++) {
            if (cfg->roles[i].eps[j].cap.known) continue;   /* уже знаем */
            args[n].cfg = cfg;
            args[n].role_idx = i;
            args[n].ep_idx = j;
            if (pthread_create(&th[n], NULL, probe_one, &args[n]) == 0) n++;
            else
                probe_one(&args[n]);
        }
    for (int k = 0; k < n; k++) pthread_join(th[k], NULL);
}

/* ------------------------------------------------------- пул рабочих --- */
/*
 * Рабочий роль больше не объявляет: в объявлении приходит только мощь —
 * сколько байт под веса узел отдаёт, какой лимит тела и на какой платформе.
 * Кем рабочий будет и сколько он сделает, решает распределитель, причём
 * отдельно на каждую задачу: роль уже стоит в теле задания, и рабочий её
 * исполняет.
 *
 * Поэтому найденных рабочих складываем в один пул ([role *]) и берём
 * исполнителя оттуда, когда конкретная роль в конфигурации не описана.
 * Перечисленные в конфигурации узлы по-прежнему приоритетнее — так можно
 * молча оставить конкретную машину на конкретной работе.
 */

static pthread_mutex_t pool_mu = PTHREAD_MUTEX_INITIALIZER;

/* Уже в пуле? Ключ — имя и рабочий порт узла, а не адрес пакета: один и тот
 * же узел слышим и по 127.0.0.1, и по адресу интерфейса. Без такой сверки
 * в пуле окажется копия каждого устройства, и раскладка модели попытается
 * положить два слоя на одну машину. */
static int pool_seen(const dc_config *cfg, const char *url, const char *node, int port)
{
    dc_role *r = conf_role((dc_config *)cfg, ROLE_POOL, 0);
    if (!r || !node || !*node) return 0;
    for (int i = 0; i < r->nep; i++) {
        if (strcmp(r->eps[i].url, url) == 0) return 1;
        if (strcmp(r->eps[i].cap.node, node) != 0) continue;
        char host[256] = {0};
        int  p = 0;
        char eperr[128] = {0};
        if (udp_parse_endpoint(r->eps[i].url, host, sizeof host, &p, eperr, sizeof eperr) == 0 &&
            p == port)
            return 1;
    }
    return 0;
}

/*
 * Находит рабочих в сети и складывает их в пул. Рабочий отвечает на запрос
 * объявлением немедленно, поэтому ждать объявлений по таймеру не нужно, но
 * сам запрос повторяем каждые полсекунды: объявление могло пройти мимо.
 * Возвращает число новых рабочих, либо -1, если порт discoveries занят.
 */
int sched_pool_fill(dc_config *cfg, int seconds, char *err, size_t errlen)
{
    if (seconds <= 0) seconds = 2;
    if (seconds > 30) seconds = 30;

    char derr[256] = {0};
    int  fd = udp_socket_bound("0.0.0.0", DC_DISCOVERY_PORT, derr, sizeof derr);
    if (fd < 0) {
        if (err && errlen) snprintf(err, errlen, "%s", derr);
        return -1;
    }

    int on = 1;
    setsockopt(fd, SOL_SOCKET, SO_BROADCAST, &on, sizeof on);
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);

    int ask = udp_socket_client(NULL, 0);
    /* без SO_BROADCAST широковещание уходит в EACCES — и молча */
    if (ask >= 0) setsockopt(ask, SOL_SOCKET, SO_BROADCAST, &on, sizeof on);

    char ask_json[] = "{\"role\":\"discover\"}";
    struct sockaddr_in baddr;
    memset(&baddr, 0, sizeof baddr);
    baddr.sin_family = AF_INET;
    baddr.sin_port   = htons(DC_DISCOVERY_PORT);

    long deadline = now_ms() + (long)seconds * 1000;
    long next_ask = 0;
    long first_at = 0;
    int  found = 0;

    while (now_ms() < deadline) {
        /* Узел отвечает на запрос сразу, поэтому ждать всё окно незачем:
         * как только первый ответ пришёл, даём подтянуться остальным и
         * выходим. Иначе панель, ради которой и ищем рабочих, встала бы
         * на лишние секунды. */
        if (found > 0 && now_ms() - first_at > 400) break;

        if (ask >= 0 && now_ms() >= next_ask) {
            unsigned char *d = NULL;
            size_t         dl = 0;
            if (udp_pack(DC_MSG_PING, ask_json, &d, &dl) == 0) {
                static const unsigned long bcast[2] = { 0xFFFFFFFFu, 0x7FFFFFFFu };
                for (int i = 0; i < 2; i++) {
                    baddr.sin_addr.s_addr = htonl(bcast[i]);
                    if (sendto(ask, d, dl, 0, (struct sockaddr *)&baddr, sizeof baddr) < 0)
                        log_debug("широковещательный запрос не ушёл: %s", strerror(errno));
                }
            }
            free(d);
            next_ask = now_ms() + 500;
        }

        unsigned char      buf[2048];
        struct sockaddr_in from;
        socklen_t          flen = sizeof from;
        struct timeval     tv = { 0, 200000 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

        ssize_t r = recvfrom(fd, buf, sizeof buf, 0, (struct sockaddr *)&from, &flen);
        if (r <= 0) continue;

        dc_msg m;
        char   perr[192] = {0};
        if (udp_unpack(buf, (size_t)r, &m, perr, sizeof perr) != 0) continue;
        if (m.type != DC_MSG_HELLO) continue;

        char json[2049];
        size_t len = m.body_len < sizeof json - 1 ? m.body_len : sizeof json - 1;
        memcpy(json, m.body, len);
        json[len] = '\0';

        js_doc *doc = js_parse(json, perr, sizeof perr);
        if (!doc) continue;
        const js_value *root = js_root(doc);
        const char *node = js_get_str(root, "node");
        char        nm[32];
        snprintf(nm, sizeof nm, "%s", node ? node : "");
        /* порт берём из объявления: источник HELLO — эфемерный сокет */
        int         port = (int)js_num(js_get(root, "port"), 0);
        js_doc_free(doc);
        if (port <= 0) continue;

        char url[160];
        snprintf(url, sizeof url, "udp://%s:%d", inet_ntoa(from.sin_addr), port);

        pthread_mutex_lock(&pool_mu);
        int dup = pool_seen(cfg, url, nm, port);
        int rc = 0;
        if (!dup) {
            dc_role *pool = conf_role(cfg, ROLE_POOL, 1);
            if (!pool) rc = -1;
            else {
                int before = pool->nep;
                rc = conf_role_add_ep(pool, url);
                /* имя узла запомним сразу: по нему же узнаём дубликат, когда
                 * тот же узел объявится со второго адреса. Точные возможности
                 * проставит общий опрос по PING. */
                if (rc == 0 && pool->nep > before)
                    snprintf(pool->eps[before].cap.node, sizeof pool->eps[before].cap.node,
                             "%s", nm);
            }
        }
        pthread_mutex_unlock(&pool_mu);

        if (!dup && rc == 0) {
            if (!found) first_at = now_ms();
            found++;
        }
    }

    close(fd);
    if (ask >= 0) close(ask);
    return found;
}

/* ------------------------------------------------- выбор узла ------- */

/* Узел подходит, если принимает тело заданного размера. Роль рабочему больше
 * ничего не говорит: её назначает распределитель в теле задания, поэтому
 * исполнителя берём из общей секции роли, а если та не описана — из пула
 * найденных рабочих. Для маленьких устройств max_body мал, и это главный
 * фильтр: без него планировщик слал бы ESP32 задачу на 8 КБ и молча ждал
 * таймаут.
 *
 * Если у шага закреплён узел (pin) — выбор один: слой модели лежит там
 * и только там, отдать его соседу нельзя. */
static dc_endpoint *pick_ep(runner *r, const char *role, const char *pin, int body_len)
{
    dc_role *rl = conf_role(r->cfg, role, 0);
    if (!rl || rl->nep == 0) rl = conf_role(r->cfg, ROLE_POOL, 0);
    if (!rl || rl->nep == 0) return NULL;

    lock(r);
    dc_endpoint *best = NULL;
    if (pin && *pin) {
        for (int j = 0; j < rl->nep; j++)
            if (strcmp(rl->eps[j].url, pin) == 0) {
                best = &rl->eps[j];
                /* не влезает даже туда, куда закреплено, — тоже мимо */
                if (best->cap.known && best->cap.max_body > 0 && body_len > best->cap.max_body)
                    best = NULL;
                break;
            }
    } else {
        int start = (int)(r->rr % rl->nep);
        for (int k = 0; k < rl->nep; k++) {
            dc_endpoint *e = &rl->eps[(start + k) % rl->nep];
            if (e->cap.known && e->cap.max_body > 0 && body_len > e->cap.max_body) continue;
            if (!best || e->inflight < best->inflight) best = e;
        }
    }
    if (best) best->inflight++;
    if (!(pin && *pin)) r->rr++;
    unlock(r);
    return best;
}

static void release_ep(runner *r, dc_endpoint *e, int ok)
{
    if (!e) return;
    lock(r);
    if (e->inflight > 0) e->inflight--;
    if (ok) e->completed++;
    else    e->failed++;
    unlock(r);
}

/* --------------------------------------------------- HTTP-вызов ------- */

static void build_request_body(const plan *p, int si, const job *j, strbuf *b)
{
    const plan_step *s = &p->steps[si];
    sb_add(b, "{\"role\":");
    sb_add_json_string(b, s->role);
    sb_add(b, ",\"title\":");
    sb_add_json_string(b, s->title);
    sb_add(b, ",\"payload\":");
    sb_add_json_string(b, s->payload);
    sb_printf(b, ",\"attempt\":%d", j->attempt);
    sb_add(b, ",\"context\":[");
    int first = 1;
    for (int d = 0; d < s->ndeps; d++) {
        int di = plan_step_idx(p, s->deps[d]);
        if (di < 0 || !p->steps[di].output) continue;
        if (!first) sb_add(b, ",");
        first = 0;
        sb_add(b, "{\"from\":");
        sb_add_json_string(b, p->steps[di].id);
        sb_add(b, ",\"role\":");
        sb_add_json_string(b, p->steps[di].role);
        sb_add(b, ",\"output\":");
        sb_add_json_string(b, p->steps[di].output);
        sb_add(b, "}");
    }
    sb_add(b, "]}");
}

static void *worker(void *arg)
{
    job *j = arg;

    strbuf body;
    sb_init(&body);
    build_request_body(j->p, j->step, j, &body);

    char  herr[512] = {0};
    char *resp = NULL;

    long t0 = now_ms();
    /* Транспорт выбирается по схеме адреса узла (udp://host:port, tcp://,
     * dtn://) внутри xport — здесь об этом думать не нужно. */
    int rc = xport_request(j->ep, DC_MSG_REQUEST, body.p, j->cfg->timeout_ms, &resp, herr,
                           sizeof herr);
    j->ms = now_ms() - t0;
    sb_free(&body);

    if (rc != 0) {
        j->ms = 0;
        j->ok = 0;
        j->error = xstrdup(herr);
        return NULL;
    }

    char    perr[256] = {0};
    js_doc *doc = js_parse(resp ? resp : "", perr, sizeof perr);
    free(resp);
    if (!doc) {
        j->ok = 0;
        char *m = xmalloc(640);
        snprintf(m, 640, "ответ узла не JSON: %s", perr);
        j->error = m;
        return NULL;
    }

    const js_value *root = js_root(doc);
    const js_value *okv  = js_get(root, "ok");
    int         ok  = (okv && okv->type == JS_BOOL) ? okv->bval : 1;
    const char *out = js_get_str(root, "output");
    const char *err = js_get_str(root, "error");

    /* ответ несёт возможности узла — запоминаем, чтобы дальше фильтровать
     * по размеру и не слать маленькому устройству то, что оно не примет */
    if (j->ep) {
        dc_capability *cap = &j->ep->cap;
        double          mb = js_num(js_get(root, "max_body"), 0);
        double          fm = js_num(js_get(root, "free_mem"), 0);
        double          th = js_num(js_get(root, "threads"), 0);
        const char    *nd = js_get_str(root, "node");
        const char    *rs = js_get_str(root, "roles");
        const char    *sh = js_get_str(root, "shard");
        if (mb > 0) {
            cap->max_body = (int)mb;
            cap->known = 1;
        }
        if (fm > 0) cap->free_mem = (long)fm;
        if (th > 0) cap->threads = (int)th;
        if (nd) snprintf(cap->node, sizeof cap->node, "%s", nd);
        if (rs) snprintf(cap->roles, sizeof cap->roles, "%s", rs);
        if (sh) snprintf(cap->shard, sizeof cap->shard, "%s", sh);
    }

    if (!ok) {
        j->ok = 0;
        j->error = xstrdup(err && *err ? err : "узел вернул ok=false без описания ошибки");
    } else {
        j->ok = 1;
        j->output = xstrdup(out ? out : "");
    }
    js_doc_free(doc);
    return NULL;
}

/* -------------------------------------------------------- запуск ------ */

static int launch(runner *r, int si)
{
    for (int i = 0; i < MAX_SLOTS; i++) {
        if (r->th_step[i] >= 0) continue;

        plan_step  *s = &r->p->steps[si];

        /* размер запроса известен заранее: он нужен и для выбора узла по
         * max_body, и для понятной ошибки, если не подошёл ни один. */
        strbuf probe;
        sb_init(&probe);
        {
            job probe_job;
            memset(&probe_job, 0, sizeof probe_job);
            probe_job.p = r->p;
            probe_job.step = si;
            probe_job.attempt = s->attempt + 1;
            build_request_body(r->p, si, &probe_job, &probe);
        }
        int body_len = (int)probe.len;
        sb_free(&probe);

        dc_endpoint *ep = pick_ep(r, s->role, s->pin, body_len);
        if (!ep) {
            printf("  [%s] %-4s %-8s %-24s некуда отправить: %d байт, узлы не берут\n",
                   state_icon(ST_RUNNING), s->id, s->role, "-", body_len);
            s->state = ST_FAILED;
            free(s->error);
            dc_role *rl = conf_role(r->cfg, s->role, 0);
            if (s->pin && *s->pin) {
                char m[512];
                snprintf(m, sizeof m,
                         "шаг закреплён за узлом %s, но он не взял задачу в %d байт "
                         "(у него лимит %d байт) — разбейте шаг на части поменьше",
                         s->pin, body_len,
                         rl && rl->nep ? rl->eps[0].cap.max_body : 0);
                s->error = xstrdup(m);
            } else if (rl && rl->nep > 0) {
                char m[512];
                int  lim = rl->eps[0].cap.known ? rl->eps[0].cap.max_body : 0;
                snprintf(m, sizeof m,
                         "задача %d байт не помещается ни в один узел роли %s (у них %d байт); "
                         "разбейте шаг на части поменьше",
                         body_len, s->role, lim);
                s->error = xstrdup(m);
            } else {
                s->error = xstrdup("для роли не назначен ни один узел");
            }
            return 0;
        }

        job *j = &r->jobs[i];
        memset(j, 0, sizeof *j);
        j->cfg = r->cfg;
        j->p = r->p;
        j->step = si;
        j->ep = ep;
        j->started = now_ms();

        s->state = ST_RUNNING;
        /* копия, а не указатель на адрес узла: план переживает конфигурацию,
         * из которой он был выполнен, иначе в UI появится мусор вместо адреса */
        free(s->ep);
        s->ep = xstrdup(ep->url);
        s->attempt++;

        /* печатаем здесь, а не на месте вызова: только теперь известен
         * фактически выбранный узел, а не первый в списке */
        printf("  ");
        TC(stdout, C_BLUE, "[%s]", state_icon(ST_RUNNING));
        printf(" %-4s %-8s %-24s %s (%d байт)\n", s->id, s->role, s->ep, s->title, body_len);

        if (pthread_create(&r->th[i], NULL, worker, j) != 0) {
            release_ep(r, ep, 0);
            s->state = ST_FAILED;
            free(s->error);
            s->error = xstrdup("не удалось создать рабочий поток");
            return 0;
        }
        r->th_step[i] = si;
        return 1;
    }
    return 0;
}

/* применяет результат потока к шагу; 1 — шаг в терминальном состоянии,
 * 0 — будет повторён */
static int apply_result(runner *r, int slot, int si, int max_attempts)
{
    job        *j = &r->jobs[slot];
    plan_step  *s = &r->p->steps[si];

    s->ms = j->ms;

    if (j->ok) {
        s->state = ST_DONE;
        free(s->output);
        s->output = xstrdup(j->output ? j->output : "");
        free(s->error);
        s->error = NULL;
        release_ep(r, j->ep, 1);
        printf("  ");
        TC(stdout, C_GREEN, "[%s]", state_icon(s->state));
        printf(" %-4s %-8s %-30s -> ", s->id, s->role, s->ep);
        TC(stdout, C_BOLD, "%s", j->output && *j->output ? j->output : "пусто");
        printf("  ");
        TC(stdout, C_DIM, "(%ld мс)", j->ms);
        putchar('\n');
        return 1;
    }

    release_ep(r, j->ep, 0);
    free(s->output);
    s->output = NULL;
    free(s->error);
    s->error = xstrdup(j->error ? j->error : "неизвестная ошибка");

    if (s->attempt < max_attempts) {
        long backoff = r->cfg->backoff_ms * s->attempt;
        if (backoff < 0) backoff = 0;
        printf("  ");
        TC(stdout, C_YELLOW, "[%s]", state_icon(ST_RUNNING));
        printf(" %-4s %-8s %-30s попытка %d не удалась: %s — повтор через %ld мс\n", s->id,
               s->role, s->ep, s->attempt, s->error, backoff);
        s->state = ST_PENDING;
        return 0;
    }

    s->state = ST_FAILED;
    printf("  ");
    TC(stdout, C_RED, "[%s]", state_icon(s->state));
    printf(" %-4s %-8s %-30s ОШИБКА: %s\n", s->id, s->role, s->ep ? s->ep : "-", s->error);
    return 1;
}

static void job_reset(job *j)
{
    free(j->output);
    free(j->error);
    j->output = NULL;
    j->error = NULL;
}

/* --------------------------------------------------------- запуск ----- */

int sched_run(plan *p, dc_config *cfg)
{
    runner r;
    memset(&r, 0, sizeof r);
    r.cfg = cfg;
    r.p = p;
    pthread_mutex_init(&r.mu, NULL);
    for (int i = 0; i < MAX_SLOTS; i++) r.th_step[i] = -1;

    int concurrency = cfg->concurrency > 0 ? cfg->concurrency : 4;
    int max_attempts = 1 + (cfg->retries > 0 ? cfg->retries : 0);

    /* Узнаём, что каждый узел способен принять, до того как начнём слать.
     * Без этого планировщик отправил бы шаг на 8 КБ маленькой плате, таймаут
     * истёк бы, и виноват был бы «мёртвый узел», хотя он жив. */
    probe_all(cfg);

    /* отложенные повторы: время, раньше которого шаг запускать нельзя */
    long *ready_at = xcalloc((size_t)p->n, sizeof *ready_at);

    int running = 0, terminal = 0;
    printf("Запуск плана (%d шагов, параллелизм %d, попыток на шаг %d)\n\n", p->n,
           concurrency, max_attempts);

    while (terminal < p->n) {
        long now = now_ms();

        /* 1. шаги, чьи зависимости провалились, помечаем заблокированными */
        for (int i = 0; i < p->n; i++) {
            plan_step *s = &p->steps[i];
            if (s->state != ST_PENDING) continue;
            int blocked = 0;
            for (int d = 0; d < s->ndeps; d++) {
                int di = plan_step_idx(p, s->deps[d]);
                if (di < 0) continue;
                step_state ds = p->steps[di].state;
                if (ds == ST_FAILED || ds == ST_BLOCKED) blocked = 1;
            }
            if (blocked) {
                s->state = ST_BLOCKED;
                free(s->error);
                s->error = xstrdup("зависимость не выполнена");
                terminal++;
                printf("  ");
                TC(stdout, C_YELLOW, "[%s]", state_icon(s->state));
                printf(" %-4s %-8s %-30s пропущен (зависимость упала)\n", s->id, s->role, "-");
            }
        }

        /* 2. запускаем всё готовое */
        for (int i = 0; i < p->n && running < concurrency; i++) {
            plan_step *s = &p->steps[i];
            if (s->state != ST_PENDING || ready_at[i] > now) continue;
            int ready = 1;
            for (int d = 0; d < s->ndeps; d++) {
                int di = plan_step_idx(p, s->deps[d]);
                if (di >= 0 && p->steps[di].state != ST_DONE) ready = 0;
            }
            if (!ready) continue;
            if (launch(&r, i)) running++;
            else terminal++;
        }

        if (running == 0) {
            /* ждём выхода отложенного повтора */
            long soonest = -1;
            for (int i = 0; i < p->n; i++) {
                if (p->steps[i].state != ST_PENDING) continue;
                if (ready_at[i] > now && (soonest < 0 || ready_at[i] < soonest))
                    soonest = ready_at[i];
            }
            if (soonest > 0) {
                long wait = soonest - now;
                usleep((useconds_t)(wait > 200 ? 200 : wait) * 1000);
                continue;
            }
            /* запускать больше нечего — оставшиеся шаги неисполнимы */
            int stalled = 0;
            for (int i = 0; i < p->n; i++) {
                if (p->steps[i].state != ST_PENDING) continue;
                p->steps[i].state = ST_BLOCKED;
                free(p->steps[i].error);
                p->steps[i].error = xstrdup("шаг не может быть запущен");
                terminal++;
                stalled = 1;
                printf("  ");
                TC(stdout, C_YELLOW, "[%s]", state_icon(ST_BLOCKED));
                printf(" %-4s %-8s %-30s не запускается\n", p->steps[i].id, p->steps[i].role, "-");
            }
            if (!stalled) break;
            continue;
        }

        /* 3. ждём любого завершившегося потока */
        int slot = -1;
        for (int i = 0; i < MAX_SLOTS; i++)
            if (r.th_step[i] >= 0) { slot = i; break; }
        if (slot < 0) break;

        pthread_join(r.th[slot], NULL);
        r.th_step[slot] = -1;
        running--;

        int si = r.jobs[slot].step;
        int attempts_used = r.jobs[slot].attempt;
        if (apply_result(&r, slot, si, max_attempts)) {
            terminal++;
        } else {
            ready_at[si] = now_ms() + cfg->backoff_ms * attempts_used;
            if (ready_at[si] < 0) ready_at[si] = now_ms();
        }
        job_reset(&r.jobs[slot]);
    }

    free(ready_at);
    pthread_mutex_destroy(&r.mu);

    int ok = 1;
    for (int i = 0; i < p->n; i++)
        if (p->steps[i].state != ST_DONE) ok = 0;
    return ok ? 0 : -1;
}

/* --------------------------------------------------- печать плана ----- */

void sched_print_plan(const plan *p, const dc_config *cfg)
{
    printf("Цель: ");
    TC(stdout, C_BOLD, "%s", p->goal);
    printf("\n\n  ");
    TC(stdout, C_DIM, "%-4s %-9s %-12s %-30s %s", "id", "роль", "зависит", "узел", "задача");
    putchar('\n');
    for (int oi = 0; oi < p->norder; oi++) {
        int              i = p->order[oi];
        const plan_step *s = &p->steps[i];

        char deps[64] = "-";
        if (s->ndeps > 0) {
            snprintf(deps, sizeof deps, "%s", s->deps[0]);
            for (int d = 1; d < s->ndeps && d < 3; d++) {
                strncat(deps, ",", sizeof deps - strlen(deps) - 1);
                strncat(deps, s->deps[d], sizeof deps - strlen(deps) - 1);
            }
            if (s->ndeps > 3) strncat(deps, ",...", sizeof deps - strlen(deps) - 1);
        }

        const dc_role *own = conf_role((dc_config *)cfg, s->role, 0);
        int from_pool = (!own || own->nep == 0);
        const dc_role *r = from_pool ? conf_role((dc_config *)cfg, ROLE_POOL, 0) : own;
        char eps[160] = "НЕ НАЗНАЧЕН";
        if (s->pin) {
            /* закреплённый шаг показываем на том узле, где он и пойдёт */
            snprintf(eps, sizeof eps, "%s (закреплён)", s->pin);
            printf("  %-4s %-9s %-12s %-30s %s\n", s->id, s->role, deps, eps, s->title);
            continue;
        }
        if (r && r->nep) {
            snprintf(eps, sizeof eps, "%s", r->eps[0].url);
            if (r->nep > 1) {
                char num[24];
                /* роль никто не объявлял: исполнителя берём из пула целиком */
                snprintf(num, sizeof num, from_pool ? " (+%d из пула)" : " (+%d)", r->nep - 1);
                strncat(eps, num, sizeof eps - strlen(eps) - 1);
            } else if (from_pool) {
                strncat(eps, " из пула", sizeof eps - strlen(eps) - 1);
            }
        }
        printf("  %-4s %-9s %-12s %-30s %s\n", s->id, s->role, deps, eps, s->title);
    }
    printf("\n");
}

/* ------------------------------------------------------- проверка ----- */
/*
 * Один PING, из которого получаем и задержку, и возможности узла.
 * Раньше здесь было два обмена: сначала ping ради задержки, потом второй
 * ради PONG — лишний круг на каждый узел при каждой проверке.
 *
 * Транспорт определяется по схеме адреса узла: udp://, tcp://, dtn://.
 */
static int node_probe(dc_config *cfg, dc_endpoint *ep, int *ms_out, dc_capability *cap,
                      char *err, size_t errlen)
{
    char  perr[512] = {0};
    char *pong = NULL;

    /* Если у узла есть параметры в конфигурации — добавляем их в PING,
     * чтобы узел мог применить их (например, max_body или hello_interval) */
    char ping_json[512];
    /* threads в PING всегда: -1 означает «не трогай», 0 — «все ядра»,
     * а узлу тяжелее, чем самому угадывать, сколько ему считать. */
    int cfg_thr = -1;
    if (ep) {
        const dc_node_params *np = conf_effective_params(cfg, ep);
        cfg_thr = np->threads;
    }
    if (ep && (ep->params.max_body > 0 || ep->params.hello_secs > 0 ||
               ep->params.use_tcp || ep->params.use_dtn || cfg_thr >= 0)) {
        snprintf(ping_json, sizeof ping_json,
                 "{\"role\":\"ping\",\"cfg_max_body\":%d,\"cfg_free_mem\":%ld,"
                 "\"cfg_hello_secs\":%d,\"cfg_use_tcp\":%d,\"cfg_use_dtn\":%d,"
                 "\"cfg_threads\":%d}",
                 ep->params.max_body, ep->params.free_mem,
                 ep->params.hello_secs, ep->params.use_tcp, ep->params.use_dtn, cfg_thr);
    } else {
        snprintf(ping_json, sizeof ping_json,
                 "{\"role\":\"ping\",\"cfg_threads\":%d}", cfg_thr);
    }

    long t0 = now_ms();
    int  rc = xport_request(ep, DC_MSG_PING, ping_json, 3000, &pong, perr, sizeof perr);
    if (ms_out) *ms_out = (int)(now_ms() - t0);

    if (rc != 0) {
        snprintf(err, errlen, "%s", perr);
        return -1;
    }

    char    jerr[192] = {0};
    js_doc *d = js_parse(pong ? pong : "", jerr, sizeof jerr);
    free(pong);
    if (!d) {
        snprintf(err, errlen, "PONG без разбора: %s", jerr);
        return -1;
    }

    if (cap) {
        /* max_body приходит числом, поэтому js_get_str здесь вернул бы NULL —
         * и фильтр по размеру молча не включился бы */
        double mb = js_num(js_get(js_root(d), "max_body"), 0);
        const char *rs = js_get_str(js_root(d), "roles");
        const char *ar = js_get_str(js_root(d), "arch");
        const char *nd = js_get_str(js_root(d), "node");
        /* сколько памяти узел отдаёт под веса модели */
        double fm = js_num(js_get(js_root(d), "free_mem"), 0);
        double th = js_num(js_get(js_root(d), "threads"), 0);
        const char *sh = js_get_str(js_root(d), "shard");
        if (mb > 0) {
            cap->max_body = (int)mb;
            cap->known = 1;
        }
        if (fm > 0) cap->free_mem = (long)fm;
        if (th > 0) cap->threads = (int)th;
        if (rs) snprintf(cap->roles, sizeof cap->roles, "%s", rs);
        if (ar) snprintf(cap->arch, sizeof cap->arch, "%s", ar);
        if (nd) snprintf(cap->node, sizeof cap->node, "%s", nd);
        if (sh) snprintf(cap->shard, sizeof cap->shard, "%s", sh);
    }
    js_doc_free(d);
    return 0;
}

/* Узнать, что умеет каждый узел, ничего не печатая. Уже опрошенные узлы
 * пропускаются: их возможности лежат в cap и не меняются сами по себе. */
int sched_learn_caps(dc_config *cfg)
{
    if (!cfg) return 0;
    probe_all(cfg);
    int known = 0;
    for (int i = 0; i < cfg->nroles; i++)
        for (int j = 0; j < cfg->roles[i].nep; j++)
            if (cfg->roles[i].eps[j].cap.known) known++;
    return known;
}

int sched_probe(dc_config *cfg, dc_node_status *out, int max)
{
    int total = 0;
    for (int i = 0; i < cfg->nroles && total < max; i++) {
        dc_role *r = &cfg->roles[i];
        for (int j = 0; j < r->nep && total < max; j++) {
            dc_node_status *ns = &out[total++];
            memset(ns, 0, sizeof *ns);
            snprintf(ns->role, sizeof ns->role, "%s", r->name);
            snprintf(ns->url, sizeof ns->url, "%s", r->eps[j].url);

            char host[320];
            int  port = 0;
            char err[256] = {0};
            if (xport_parse(r->eps[j].url, host, sizeof host, &port, err, sizeof err) != 0) {
                ns->alive = 0;
                snprintf(ns->detail, sizeof ns->detail, "%s", err);
                continue;
            }
            /* host:port для интерфейса; собираем вручную, чтобы не ругался компилятор */
            size_t hl = strlen(host);
            if (hl + 8 < sizeof ns->host) {
                memcpy(ns->host, host, hl);
                ns->host[hl] = '\0';
                snprintf(ns->host + hl, sizeof ns->host - hl, ":%d", port);
            } else {
                snprintf(ns->host, sizeof ns->host, "%s", r->eps[j].url);
            }

            int ms = 0;
            if (node_probe(cfg, &r->eps[j], &ms, &r->eps[j].cap, err, sizeof err) != 0) {
                ns->alive = 0;
                snprintf(ns->detail, sizeof ns->detail, "%s", err);
                continue;
            }
            ns->alive = 1;
            ns->ms = ms;
            snprintf(ns->detail, sizeof ns->detail, "ответ за %d мс", ms);
            ns->max_body = r->eps[j].cap.max_body;
            ns->free_mem = r->eps[j].cap.free_mem;
            snprintf(ns->arch, sizeof ns->arch, "%s", r->eps[j].cap.arch);
            snprintf(ns->shard, sizeof ns->shard, "%s", r->eps[j].cap.shard);
        }
    }
    return total;
}

int sched_check(dc_config *cfg)
{
    int bad = 0, total = 0;
    for (int i = 0; i < cfg->nroles; i++) {
        dc_role *r = &cfg->roles[i];
        /* пул — не роль: это просто найденные рабочие, роль им назначит
         * распределитель, когда дойдёт дело */
        int is_pool = r->name && strcmp(r->name, ROLE_POOL) == 0;
        if (is_pool) printf("пул            %d узел(ов)\n", r->nep);
        else          printf("роль %-9s %d узел(ов)\n", r->name, r->nep);
        if (r->nep == 0) {
            if (is_pool)
                printf("    рабочие не найдены — запустите dcd, распределитель найдёт их сам\n");
            else
                printf("    узлы не назначены — добавьте [role %s] daemon = host:port\n", r->name);
            bad++;
            continue;
        }
        for (int j = 0; j < r->nep; j++) {
            total++;
            char host[320];
            int  port = 0;
            char err[256] = {0};
            if (xport_parse(r->eps[j].url, host, sizeof host, &port, err, sizeof err) != 0) {
                printf("    %s:%d", host, port);
                printf("  некорректный адрес: %s\n", err);
                bad++;
                continue;
            }

            dc_capability cap;
            memset(&cap, 0, sizeof cap);
            int ms = 0;
            if (node_probe(cfg, &r->eps[j], &ms, &cap, err, sizeof err) != 0) {
                printf("    %s", r->eps[j].url);
                printf("  недоступен: %s\n", err);
                bad++;
                continue;
            }
            /* показываем, что за устройство перед нами и на что оно способно */
            printf("    %s  ", r->eps[j].url);
            printf("живой, %3d мс", ms);
            if (cap.known) printf(", лимит тела %d байт", cap.max_body);
            if (cap.threads > 0) {
                int a = cap.threads % 10, b = cap.threads % 100;
                const char *w = (a == 1 && b != 11) ? "поток"
                           : (a >= 2 && a <= 4 && !(b >= 12 && b <= 14)) ? "потока"
                           : "потоков";
                printf(", %d %s", cap.threads, w);
            }
            if (cap.free_mem > 0) printf(", под модель %ld байт", cap.free_mem);
            if (cap.shard[0] && strncmp(cap.shard, "пусто", 5) != 0) printf(" [%s]", cap.shard);
            if (cap.roles[0] && strcmp(cap.roles, "*") != 0)
                printf(", роли: %s", cap.roles);
            else
                printf(", роль любая");
            if (cap.arch[0]) printf(" [%s]", cap.arch);
            putchar('\n');
        }
    }
    printf("\nпроверено %d узлов, проблем %d\n", total, bad);
    return bad == 0 ? 0 : -1;
}
