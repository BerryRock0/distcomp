#include "web.h"
#include "json.h"
#include "llm.h"
#include "plan.h"
#include "sched.h"
#include "util.h"

#include <arpa/inet.h>
#include <dirent.h>
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

/* 64 МБ: через тело запроса приходят и загружаемые файлы, а не только JSON.
 * Буфер под тело выделяется по фактическому Content-Length, так что размер
 * влияет только на потолок одного запроса, а не на постоянную память. */
#define MAX_BODY   (64 << 20)

/* Чат держит соединение минутами: модель загружается десятки секунд, а ответ
 * идёт по одному токену за шаг конвейера через сеть. Обычный таймаут в 30 с
 * обрывал бы поток на середине. */
#define DC_CHAT_TIMEOUT_S 1800
#define MAX_RUNS   64

/* Сколько файлов моделей максимум отдаём в панель: каталог с моделями —
 * обычно десяток файлов, а память под список всё равно надо где-то брать */
#define DC_MAX_MODEL_FILES 100

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

    /* Чат. Модель разложена по узлам один раз и живёт здесь: разложение и
     * заливка весов занимают десятки секунд, а перезагружать её на каждое
     * сообщение нельзя. Генерация идёт по одному потоку — состояние конвейера
     * общее, поэтому chat_mu держит и сам конвейер, и историю разговора. */
    dc_infer        *chat_infer;
    pthread_mutex_t  chat_mu;
    strbuf           chat_hist;        /* реплики в формате ChatML */
    int              chat_ready;       /* 1 — модель загружена и готова */
    int              chat_busy;        /* 1 — идёт генерация */
    char             chat_err[512];    /* почему не загрузилась */

    /* Каталог загруженных файлов. Один мьютекс на все операции с ними:
     * загрузка, список, скачивание и удаление идут из разных потоков, и без
     * него список мог прочитать каталог, пока в него пишут. */
    pthread_mutex_t  up_mu;
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

/* extra — готовые дополнительные заголовки (каждая строка с \r\n), вставляются
 * перед Cache-Control. Нужно для Content-Disposition при скачивании файла. */
static void http_reply_extra(int fd, int status, const char *ctype, const char *extra,
                             const char *body, size_t len)
{
    const char *reason = status == 200 ? "OK"
                       : status == 404 ? "Not Found"
                       : status == 400 ? "Bad Request"
                       : status == 409 ? "Conflict"
                       : status == 500 ? "Internal Server Error"
                                       : "Error";
    char head[768];
    int  hn = snprintf(head, sizeof head,
                       "HTTP/1.1 %d %s\r\n"
                       "Content-Type: %s\r\n"
                       "Content-Length: %zu\r\n"
                       "%s"
                       "Cache-Control: no-store\r\n"
                       "Connection: close\r\n\r\n",
                       status, reason, ctype, len, extra ? extra : "");
    /* snprintf вернул бы «длину без обрезки»: без клампа write читал бы
     * за пределами стекового буфера */
    if (hn < 0 || (size_t)hn >= sizeof head) hn = (int)sizeof head - 1;
    if (write(fd, head, (size_t)hn) < 0) return;
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, body + off, len - off);
        /* Прерывание по сигналу — это не отказ: write не перезапрашивает
         * себя сам, и если на этом разорвать, клиент получит обрезанный ответ.
         * В частности отваливался элемент в конце страницы панели: заголовок
         * успевал уйти, хвост — нет. */
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) break;
        off += (size_t)w;
    }
}

static void http_reply(int fd, int status, const char *ctype, const char *body, size_t len)
{
    http_reply_extra(fd, status, ctype, "", body, len);
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

/* «настроена» = локальная модель разложилась по узлам роли shard и планы
 * можно строить без внешнего сервера */
static int llm_available_for_web(const dc_config *cfg)
{
    llm_status st;
    return llm_local_status(cfg, &st);
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
        /* вид узла: тяжёлый держит модель, лёгкий считает только выражения.
         * Панель показывает это меткой, иначе «свободно 0 байт» выглядит
         * как поломка, а это просто устройство, которому модель не нужна. */
        sb_add(&b, ",\"flavour\":");
        sb_add_json_string(&b, ns[i].flavour[0] ? ns[i].flavour : "heavy");
        sb_add(&b, ",\"detail\":");
        sb_add_json_string(&b, ns[i].detail);
        sb_add(&b, "}");
    }
    sb_add(&b, "],\"settings\":{");
    sb_printf(&b, "\"llm_model\":");
    sb_add_json_string(&b, w->cfg->model_file);
    /* «настроена» = модель разложилась по узлам, а не просто файл существует:
     * иначе панель врала бы о парке, который модель не вмещает */
    sb_printf(&b, ",\"llm_configured\":%s", llm_available_for_web(w->cfg) ? "true" : "false");
    sb_printf(&b, ",\"llm_n_ctx\":%d", w->cfg->n_ctx);
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

/* Проверка локальной модели по кнопке в панели: модель разложена по узлам?
 * Отвечает тот же llm_local_status, что и CLI, но его вывод идёт в ответ,
 * а не в терминал панели. */
static void api_llm(dc_web *w, int fd)
{
    dc_config *cfg = NULL;
    pthread_mutex_lock(&w->mu);
    cfg = conf_clone(w->cfg);
    pthread_mutex_unlock(&w->mu);

    strbuf     b;
    sb_init(&b);
    llm_status st;
    memset(&st, 0, sizeof st);
    int ready = cfg ? llm_local_status(cfg, &st) : 0;

    sb_add(&b, "{\"ok\":");
    sb_add(&b, ready ? "true" : "false");
    sb_add(&b, ",\"error\":");
    sb_add_json_string(&b, ready ? "" : st.note);
    sb_add(&b, ",\"model\":");
    sb_add_json_string(&b, st.model);
    sb_add(&b, ",\"arch\":");
    sb_add_json_string(&b, st.arch);
    sb_printf(&b, ",\"n_layer\":%d,\"n_embd\":%d,\"n_vocab\":%d", st.n_layer, st.n_embd,
              st.n_vocab);
    sb_printf(&b, ",\"n_ctx\":%d,\"n_nodes\":%d,\"n_stages\":%d", st.n_ctx, st.n_nodes,
              st.n_stages);
    sb_printf(&b, ",\"bytes\":%zu}", st.bytes);
    conf_free(cfg);
    reply_json(fd, ready ? 200 : 503, &b);
    sb_free(&b);
}

/* --------------------------------------------------------- модели ------ */

typedef struct {
    char *path;    /* путь, который уйдёт обратно в model_file */
    char *name;    /* имя файла — то, что человек видит в списке */
    char *real;    /* канонический путь — для отсева дубликатов */
    long  bytes;
} model_rec;

static int model_cmp(const void *a, const void *b)
{
    return strcmp(((const model_rec *)a)->name, ((const model_rec *)b)->name);
}

/* .gguf в любом регистре */
static int is_gguf_name(const char *name)
{
    size_t n = strlen(name);
    return n >= 5 && strncasecmp(name + n - 5, ".gguf", 5) == 0;
}

/* Один каталог в список: только регулярные *.gguf. Каталог текущей модели
 * и каталог демона частично пересекаются (модель обычно лежит рядом с
 * distcomp.conf), поэтому дубликаты отсекаем по каноническому пути — иначе
 * в списке панели дважды была бы одна и та же модель. */
static void scan_model_dir(const char *dir, model_rec **list, int *n, int *cap)
{
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d))) {
        if (!is_gguf_name(de->d_name)) continue;
        char path[2048];
        if (strcmp(dir, ".") != 0)
            snprintf(path, sizeof path, "%s/%s", dir, de->d_name);
        else
            snprintf(path, sizeof path, "%s", de->d_name);
        struct stat st;
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) continue;
        char real[2048];
        if (!realpath(path, real)) continue;
        int dup = 0;
        for (int i = 0; i < *n; i++)
            if (strcmp((*list)[i].real, real) == 0) { dup = 1; break; }
        if (dup) continue;
        if (*n >= *cap) {
            if (*cap >= DC_MAX_MODEL_FILES) continue;
            *cap *= 2;
            if (*cap > DC_MAX_MODEL_FILES) *cap = DC_MAX_MODEL_FILES;
            *list = xrealloc(*list, (size_t)*cap * sizeof **list);
        }
        model_rec *m = &(*list)[(*n)++];
        m->path  = xstrdup(path);
        m->name  = xstrdup(de->d_name);
        m->real  = xstrdup(real);
        m->bytes = (long)st.st_size;
    }
    closedir(d);
}

/* GET /api/models — список GGUF-файлов для селектора в панели. Ищем в
 * каталоге, из которого запущен демон, и в каталоге текущей модели: в
 * конфиге model_file может быть абсолютным путём в другое место. */
static void api_models(dc_web *w, int fd)
{
    char modeldir[1024] = "";
    const char *curmodel = NULL;
    pthread_mutex_lock(&w->mu);
    curmodel = w->cfg->model_file;
    if (curmodel && *curmodel && strchr(curmodel, '/')) {
        snprintf(modeldir, sizeof modeldir, "%s", curmodel);
        char *slash = strrchr(modeldir, '/');
        *slash = '\0';
    }
    pthread_mutex_unlock(&w->mu);

    int  cap = 16, n = 0;
    model_rec *list = xcalloc((size_t)cap, sizeof *list);

    if (*modeldir) scan_model_dir(modeldir, &list, &n, &cap);
    scan_model_dir(".", &list, &n, &cap);
    qsort(list, (size_t)n, sizeof *list, model_cmp);

    char curreal[2048] = "";
    if (curmodel && *curmodel && !realpath(curmodel, curreal)) curreal[0] = '\0';

    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\"current\":");
    sb_add_json_string(&b, curmodel ? curmodel : "");
    sb_add(&b, ",\"models\":[");
    for (int i = 0; i < n; i++) {
        if (i) sb_add(&b, ",");
        int iscur = *curreal && strcmp(list[i].real, curreal) == 0;
        sb_add(&b, "{\"name\":");
        sb_add_json_string(&b, list[i].name);
        sb_add(&b, ",\"path\":");
        sb_add_json_string(&b, list[i].path);
        sb_printf(&b, ",\"bytes\":%ld", list[i].bytes);
        sb_printf(&b, ",\"current\":%s}", iscur ? "true" : "false");
    }
    sb_add(&b, "]}");
    reply_json(fd, 200, &b);
    sb_free(&b);
    for (int i = 0; i < n; i++) {
        free(list[i].path);
        free(list[i].name);
        free(list[i].real);
    }
    free(list);
}

/* POST /api/model — смена модели из панели. Меняется живой model_file:
 * следующий запуск плана и следующая загрузка чата берут новый файл. На
 * диск не пишем — как и правки ролей, это действует до перезапуска демона.
 *
 * Чат при смене разгружается: веса лежат на узлах, и перезаливать их
 * нужно уже для другой модели. Пока идёт генерация, менять модель
 * нельзя — разгружать конвейер из-под текущего ответа опасно. */
static void api_model(dc_web *w, int fd, const char *body)
{
    char     perr[256] = {0};
    js_doc  *doc = js_parse(body && *body ? body : "{}", perr, sizeof perr);
    if (!doc) {
        reply_error(fd, 400, perr);
        return;
    }
    /* копируем до освобождения документа: строка живёт внутри js_doc */
    const char *mval = js_get_str(js_root(doc), "model");
    char *model = xstrdup(mval ? mval : "");
    js_doc_free(doc);
    if (!*model) {
        free(model);
        reply_error(fd, 400, "не указан файл модели");
        return;
    }
    if (strlen(model) >= 2048) {
        reply_error(fd, 400, "слишком длинный путь");
        return;
    }
    if (access(model, R_OK) != 0) {
        char msg[2200];
        snprintf(msg, sizeof msg, "файл недоступен: %s", model);
        reply_error(fd, 400, msg);
        return;
    }

    /* Смена model_file и разгрузка чата — в одном критическом разделе на
     * chat_mu: между проверкой «не идёт ли генерация» и освобождением
     * конвейера не должно быть окна — иначе генерация может начаться на
     * конвейере, который мы тут же освободим. Пока идёт генерация, модель
     * не меняем вообще: пусть ответ досчитается на старой, а панель
     * повторит выбор после него.
     *
     * mu берём вложенно (chat_mu → mu). Обратного порядка в коде нет:
     * все пути держат эти мьютексы раздельно, поэтому дедлока не будет. */
    pthread_mutex_lock(&w->chat_mu);
    if (w->chat_busy) {
        pthread_mutex_unlock(&w->chat_mu);
        reply_error(fd, 409, "идёт генерация ответа — смените модель после неё");
        return;
    }
    pthread_mutex_lock(&w->mu);
    free(w->cfg->model_file);
    w->cfg->model_file = xstrdup(model);
    pthread_mutex_unlock(&w->mu);
    dc_infer_free(w->chat_infer);
    w->chat_infer = NULL;
    w->chat_ready = 0;
    w->chat_err[0] = '\0';
    sb_free(&w->chat_hist);
    sb_init(&w->chat_hist);
    pthread_mutex_unlock(&w->chat_mu);
    log_info("модель сменена из панели: %s", model);

    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\"ok\":true,\"model\":");
    sb_add_json_string(&b, model);
    sb_add(&b, "}");
    reply_json(fd, 200, &b);
    sb_free(&b);
    free(model);
}

/* -------------------------------------------------------- файлы ------- */

/* Создаёт каталог вместе с родителями (как mkdir -p). */
static void mkdirs(const char *path)
{
    char buf[1024];
    snprintf(buf, sizeof buf, "%s", path);
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(buf, 0775);
            *p = '/';
        }
    }
    mkdir(buf, 0775);
}

/* Имя файла — только basename без путей. Символы '.' и '..' запрещены,
 * как и любые разделители путей — иначе ../etc/passwd пройдёт как имя. */
static int safe_filename(const char *name)
{
    if (!name || !*name) return 0;
    if (strchr(name, '/') || strchr(name, '\\')) return 0;
    if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) return 0;
    if (strlen(name) > 200) return 0;
    for (const unsigned char *p = (const unsigned char *)name; *p; p++)
        if (*p < 0x20 || *p == 0x7F) return 0;
    return 1;
}

/* Разбирает multipart/form-body (одна часть с файлом). Возвращает имя файла и
 * указатель на содержимое части в теле запроса (не копируем). */
static int parse_upload(const char *body, size_t len, const char *ctype,
                        char *fname, size_t fsz,
                        const char **data, size_t *dlen)
{
    /* граница из Content-Type: multipart/form-data; boundary=xxx */
    const char *b = strstr(ctype, "boundary=");
    if (!b) return -1;
    b += 9;
    char boundary[256];
    size_t bl = 0;
    while (b[bl] && b[bl] != ';' && b[bl] != '\r' && b[bl] != '\n' && bl + 1 < sizeof boundary)
        boundary[bl] = b[bl], bl++;
    boundary[bl] = '\0';
    if (!bl) return -1;

    /* разделитель в теле: "--boundary" */
    char delim[280];
    int dl = snprintf(delim, sizeof delim, "--%s", boundary);
    if (dl < 0 || (size_t)dl >= sizeof delim) return -1;
    if (len < (size_t)dl || memcmp(body, delim, (size_t)dl) != 0) return -1;

    /* пропускаем \r\n после разделителя. Проверка длины обязательна: если тело
     * кончилось ровно на разделителе, чтение h[1] ушло бы за его конец. */
    const char *h = body + dl;
    if (len - (size_t)dl >= 2 && h[0] == '\r' && h[1] == '\n') h += 2;
    else if (len - (size_t)dl >= 1 && h[0] == '\n')      h += 1;

    /* конец заголовков части — пустая строка */
    size_t hlen = len - (size_t)(h - body);
    const char *sep = memmem(h, hlen, "\r\n\r\n", 4);
    if (!sep) return -1;

    /* имя файла из Content-Disposition */
    const char *fn = memmem(h, (size_t)(sep - h), "filename=\"", 10);
    if (!fn) return -1;
    fn += 10;
    size_t fnl = 0;
    while (fn[fnl] && fn[fnl] != '"' && fnl + 1 < fsz) {
        fname[fnl] = fn[fnl];
        fnl++;
    }
    fname[fnl] = '\0';
    if (fn[fnl] != '"') return -1;

    /* содержимое между пустой строкой и следующим разделителем */
    const char *content = sep + 4;
    char tail[290];
    int tl = snprintf(tail, sizeof tail, "\r\n%s", delim);
    if (tl < 0 || (size_t)tl >= sizeof tail) return -1;
    size_t rest = len - (size_t)(content - body);
    const char *end = memmem(content, rest, tail, (size_t)tl);
    if (!end) return -1;

    *data = content;
    *dlen = (size_t)(end - content);
    return 0;
}

/* POST /api/upload — загрузка одного файла (multipart, поле file). */
static void api_upload(dc_web *w, int fd, const char *body, long blen, const char *ctype)
{
    char fname[256];
    const char *data;
    size_t dlen;
    if (blen <= 0 || !ctype ||
        parse_upload(body, (size_t)blen, ctype, fname, sizeof fname, &data, &dlen) != 0) {
        reply_error(fd, 400, "не multipart/form-data или нет поля file");
        return;
    }
    if (!safe_filename(fname)) {
        reply_error(fd, 400, "недопустимое имя файла");
        return;
    }

    char dir[1024], path[2200];
    pthread_mutex_lock(&w->mu);
    snprintf(dir, sizeof dir, "%s", w->cfg->upload_dir ? w->cfg->upload_dir : "uploads");
    pthread_mutex_unlock(&w->mu);

    mkdirs(dir);
    snprintf(path, sizeof path, "%s/%s", dir, fname);

    pthread_mutex_lock(&w->up_mu);
    FILE *f = fopen(path, "wb");
    if (!f) {
        pthread_mutex_unlock(&w->up_mu);
        reply_error(fd, 500, "не удалось открыть файл для записи");
        return;
    }
    int ok = dlen == 0 || fwrite(data, 1, dlen, f) == dlen;
    fclose(f);
    if (!ok) unlink(path);
    pthread_mutex_unlock(&w->up_mu);
    if (!ok) {
        reply_error(fd, 500, "ошибка записи на диск");
        return;
    }

    log_info("файл загружен: %s (%zu байт)", path, dlen);
    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\"ok\":true,\"name\":");
    sb_add_json_string(&b, fname);
    sb_printf(&b, ",\"bytes\":%zu}", dlen);
    reply_json(fd, 200, &b);
    sb_free(&b);
}

/* Элемент списка файлов. */
typedef struct {
    char  *name;
    long   bytes;
    time_t mtime;
} file_rec;

static int file_cmp(const void *a, const void *b)
{
    return strcmp(((const file_rec *)a)->name, ((const file_rec *)b)->name);
}

/* GET /api/files — список файлов или скачивание (если передан name). */
static void api_files(dc_web *w, int fd, const char *query)
{
    char want[512] = {0};
    int downloading = query && query_param(query, "name", want, sizeof want);

    char dir[1024];
    pthread_mutex_lock(&w->mu);
    snprintf(dir, sizeof dir, "%s", w->cfg->upload_dir ? w->cfg->upload_dir : "uploads");
    pthread_mutex_unlock(&w->mu);

    if (downloading) {
        if (!safe_filename(want)) {
            reply_error(fd, 400, "недопустимое имя файла");
            return;
        }
        char path[2200];
        snprintf(path, sizeof path, "%s/%s", dir, want);

        pthread_mutex_lock(&w->up_mu);
        /* только обычный файл: каталог или fifo с тем же именем fopen откроет,
         * а ftell вернёт мусор, и панель пошлёт клиенту этот мусор */
        struct stat pst;
        FILE *f = (stat(path, &pst) == 0 && S_ISREG(pst.st_mode)) ? fopen(path, "rb") : NULL;
        if (!f) {
            pthread_mutex_unlock(&w->up_mu);
            reply_error(fd, 404, "файл не найден");
            return;
        }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *buf = xmalloc((size_t)sz + 1);
        size_t got = sz > 0 ? fread(buf, 1, (size_t)sz, f) : 0;
        fclose(f);
        pthread_mutex_unlock(&w->up_mu);

        /* Имя для Content-Disposition: убираем кавычки и управляющие. */
        char disp_name[256];
        size_t dn = 0;
        for (const char *p = want; *p && dn + 1 < sizeof disp_name; p++) {
            unsigned char c = (unsigned char)*p;
            if (c == '"' || c == '\\' || c < 0x20) continue;
            disp_name[dn++] = (char)c;
        }
        disp_name[dn] = '\0';

        /* 512, а не 320: имя файла бывает длинным, и заголовок не должен
         * обрезаться — иначе браузер не поймёт имя и сохранит файл как
         * download без расширения */
        char extra[512];
        snprintf(extra, sizeof extra,
                 "Content-Disposition: attachment; filename=\"%s\"\r\n"
                 "X-Content-Type-Options: nosniff\r\n",
                 disp_name);
        http_reply_extra(fd, 200, "application/octet-stream", extra, buf, got);
        free(buf);
        return;
    }

    /* список файлов */
    pthread_mutex_lock(&w->up_mu);
    DIR *d = opendir(dir);
    file_rec *list = NULL;
    int n = 0, cap = 16;
    if (d) {
        list = xcalloc((size_t)cap, sizeof *list);
        struct dirent *de;
        while ((de = readdir(d))) {
            if (de->d_name[0] == '.' &&
                (de->d_name[1] == '\0' || de->d_name[1] == '.')) continue;
            char path[2200];
            snprintf(path, sizeof path, "%s/%s", dir, de->d_name);
            struct stat st;
            if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) continue;
            if (n >= cap) {
                cap *= 2;
                list = xrealloc(list, (size_t)cap * sizeof *list);
            }
            list[n].name  = xstrdup(de->d_name);
            list[n].bytes = (long)st.st_size;
            list[n].mtime = st.st_mtime;
            n++;
        }
        closedir(d);
    }
    pthread_mutex_unlock(&w->up_mu);

    qsort(list, (size_t)n, sizeof *list, file_cmp);

    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\"dir\":");
    sb_add_json_string(&b, dir);
    sb_add(&b, ",\"files\":[");
    for (int i = 0; i < n; i++) {
        if (i) sb_add(&b, ",");
        sb_add(&b, "{\"name\":");
        sb_add_json_string(&b, list[i].name);
        sb_printf(&b, ",\"bytes\":%ld,\"mtime\":%ld", list[i].bytes, list[i].mtime);
        sb_add(&b, "}");
    }
    sb_add(&b, "]}");
    reply_json(fd, 200, &b);
    sb_free(&b);
    for (int i = 0; i < n; i++) free(list[i].name);
    free(list);
}

/* POST /api/files — удалить файл: {"name":"x.txt","op":"remove"}. */
static void api_file_delete(dc_web *w, int fd, const char *body)
{
    char perr[256] = {0};
    js_doc *doc = js_parse(body && *body ? body : "{}", perr, sizeof perr);
    if (!doc) {
        reply_error(fd, 400, perr);
        return;
    }
    /* копируем до освобождения документа */
    const char *nval = js_get_str(js_root(doc), "name");
    char *name = xstrdup(nval ? nval : "");
    js_doc_free(doc);
    if (!safe_filename(name)) {
        free(name);
        reply_error(fd, 400, "недопустимое имя файла");
        return;
    }

    char dir[1024], path[2200];
    pthread_mutex_lock(&w->mu);
    snprintf(dir, sizeof dir, "%s", w->cfg->upload_dir ? w->cfg->upload_dir : "uploads");
    pthread_mutex_unlock(&w->mu);
    snprintf(path, sizeof path, "%s/%s", dir, name);

    pthread_mutex_lock(&w->up_mu);
    int rc = unlink(path);
    pthread_mutex_unlock(&w->up_mu);
    free(name);
    if (rc != 0) {
        reply_error(fd, 404, "файл не найден");
        return;
    }

    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\"ok\":true}");
    reply_json(fd, 200, &b);
    sb_free(&b);
}

/* ----------------------------------------------------------------- чат --- */

/* Одно событие потока: "data: {...}\n\n". Отдельные записи нужны, чтобы
 * сообщение не разрезалось на части, даже если пришло по TCP двумя кусками. */
static int sse_send(int fd, const char *json)
{
    char line[4096];
    int  n = snprintf(line, sizeof line, "data: %s\n\n", json);
    if (n < 0) return -1;
    size_t off = 0;
    size_t len = (size_t)n < sizeof line ? (size_t)n : sizeof line - 1;
    while (off < len) {
        ssize_t w = write(fd, line + off, len - off);
        if (w <= 0) return -1;
        off += (size_t)w;
    }
    return 0;
}

static void sse_event(int fd, const char *kind, const char *text, int tok, long ms)
{
    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\"type\":");
    sb_add_json_string(&b, kind);
    if (text) {
        sb_add(&b, ",\"text\":");
        sb_add_json_string(&b, text);
    }
    if (tok >= 0) sb_printf(&b, ",\"tok\":%d", tok);
    if (ms >= 0)  sb_printf(&b, ",\"ms\":%ld", ms);
    sb_add(&b, "}");
    sse_send(fd, b.p);
    sb_free(&b);
}

/* Загрузка модели под чат. Долгая (десятки секунд: раскладка по узлам и
 * заливка весов), поэтому ответ идёт потоком: браузер сразу видит, что
 * процесс идёт, а не молчит минуту до первого байта. */
static void chat_load(dc_web *w, int fd)
{
    const char *hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "\r\n";
    if (write(fd, hdr, strlen(hdr)) < 0) return;

    char err[512] = {0};
    long t0 = now_ms();
    sse_event(fd, "start", "разкладываю модель по узлам…", -1, -1);
    dc_infer *in = llm_open_chat(w->cfg, w->cfg->max_new, err, sizeof err);
    long ms = now_ms() - t0;

    pthread_mutex_lock(&w->chat_mu);
    if (in) {
        w->chat_infer = in;
        w->chat_ready = 1;
        w->chat_err[0] = '\0';
    } else {
        w->chat_ready = 0;
        snprintf(w->chat_err, sizeof w->chat_err, "%s", err);
    }
    int ready = w->chat_ready;
    char saved[512];
    snprintf(saved, sizeof saved, "%s", w->chat_err);
    pthread_mutex_unlock(&w->chat_mu);

    if (ready) {
        sse_event(fd, "loaded", "модель загружена", -1, ms);
    } else {
        sse_event(fd, "error", saved, -1, ms);
    }
}

typedef struct {
    dc_web *w;
    int     fd;
} chat_stream;

static int chat_on_token(int tok, const char *text, void *ud)
{
    chat_stream *cs = (chat_stream *)ud;
    sse_event(cs->fd, "token", text, tok, -1);
    return !cs->w->stopping;
}

/* POST /api/chat — весь разговор приходит сюда, ответ уходит потоком. */
static void api_chat(dc_web *w, int fd, const char *body)
{
    js_doc *d = js_parse(body ? body : "{}", NULL, 0);
    const char *msg = d ? js_get_str(js_root(d), "message") : NULL;
    int max_new = 0;
    if (d) {
        const js_value *mn = js_get(js_root(d), "max_new");
        if (mn && mn->type == JS_NUM) max_new = (int)mn->num;
    }
    char message[16384] = {0};
    if (msg) snprintf(message, sizeof message, "%s", msg);
    if (d) js_doc_free(d);

    const char *hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "\r\n";
    if (write(fd, hdr, strlen(hdr)) < 0) return;

    if (!message[0]) {
        sse_event(fd, "error", "пустое сообщение", -1, -1);
        return;
    }

    pthread_mutex_lock(&w->chat_mu);
    if (!w->chat_ready) {
        char saved[512];
        snprintf(saved, sizeof saved, "%s", w->chat_err);
        int have = w->chat_infer != NULL;
        pthread_mutex_unlock(&w->chat_mu);
        if (have) {
            sse_event(fd, "error", "модель ещё не загружена", -1, -1);
        } else {
            sse_event(fd, "error",
                      saved[0] ? saved
                               : "модель не загружена: укажите model_file в [llm] "
                                 "и узлы роли shard, затем нажмите «Загрузить модель»",
                      -1, -1);
        }
        return;
    }
    dc_infer *in = w->chat_infer;
    pthread_mutex_unlock(&w->chat_mu);

    char *sys  = xstrdup(llm_chat_system(w->cfg));
    char *prompt = llm_chat_prompt(sys, w->chat_hist.p ? w->chat_hist.p : "", message);
    free(sys);

    sse_event(fd, "start", "", -1, -1);
    long t0 = now_ms();

    chat_stream cs = { .w = w, .fd = fd };
    char err[512] = {0};
    char *answer = dc_infer_generate_cb(in, prompt, max_new > 0 ? max_new : w->cfg->max_new,
                                        chat_on_token, &cs, err, sizeof err);
    long ms = now_ms() - t0;

    if (!answer) {
        sse_event(fd, "error", err, -1, ms);
        free(prompt);
        return;
    }

    /* реплики в историю — только после успешного ответа */
    pthread_mutex_lock(&w->chat_mu);
    llm_chat_history_add(&w->chat_hist, "user", message);
    llm_chat_history_add(&w->chat_hist, "assistant", answer);
    pthread_mutex_unlock(&w->chat_mu);

    free(prompt);
    sse_event(fd, "done", answer, -1, ms);
    free(answer);
}

/* POST /api/chat/reset — начать разговор заново. Модель остаётся загруженной. */
static void api_chat_reset(dc_web *w, int fd)
{
    pthread_mutex_lock(&w->chat_mu);
    sb_free(&w->chat_hist);
    sb_init(&w->chat_hist);
    pthread_mutex_unlock(&w->chat_mu);
    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\"ok\":true}");
    reply_json(fd, 200, &b);
    sb_free(&b);
}

/* GET /api/chat — состояние чата для панели. */
static void api_chat_state(dc_web *w, int fd)
{
    strbuf b;
    sb_init(&b);
    pthread_mutex_lock(&w->chat_mu);
    int ready = w->chat_ready;
    char saved[512];
    snprintf(saved, sizeof saved, "%s", w->chat_err);
    int ctx = w->chat_infer ? dc_infer_n_ctx(w->chat_infer) : 0;
    int nodes = w->chat_infer ? dc_infer_n_nodes(w->chat_infer) : 0;
    pthread_mutex_unlock(&w->chat_mu);

    sb_add(&b, "{\"ok\":true,\"ready\":");
    sb_add(&b, ready ? "true" : "false");
    sb_add(&b, ",\"error\":");
    sb_add_json_string(&b, ready ? "" : (saved[0] ? saved
                                  : "модель не загружена: укажите model_file в [llm] "
                                    "и узлы роли shard"));
    sb_printf(&b, ",\"n_ctx\":%d,\"nodes\":%d}", ctx, nodes);
    reply_json(fd, 200, &b);
    sb_free(&b);
}


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
    char content_type[256] = {0};
    for (;;) {
        char line[2048];
        if (read_line(fd, line, sizeof line) < 0) goto done;
        if (!line[0]) break;
        if (strncasecmp(line, "Content-Length:", 15) == 0)
            content_length = strtol(line + 15, NULL, 10);
        /* 13, а не 14: "Content-Type:" ровно 13 символов. С лишним байтом
         * strncasecmp сравнил бы '\0' литерала с пробелом в строке и не
         * сошёлся бы никогда — заголовок молча игнорировался. */
        else if (strncasecmp(line, "Content-Type:", 13) == 0) {
            char *v = str_trim(line + 13);
            snprintf(content_type, sizeof content_type, "%s", v);
        }
    }

    char *body = NULL;
    long  body_len = 0;   /* сколько реально пришло, а не сколько обещали */
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
        /* Обрыв на середине не должен привести к разбору того, чего нет:
         * за незаполненным хвостом лежит мусор из malloc, и он попал бы
         * в загруженный файл. */
        body_len = (long)got;
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
    char r_models[80], r_model[80], r_files[80], r_upload[80];
    char r_chat[80], r_chat_reset[96], r_chat_load[96];
    snprintf(base, sizeof base, "%s", w->api_base);
    snprintf(r_state, sizeof r_state, "%s/state", base);
    snprintf(r_run, sizeof r_run, "%s/run", base);
    snprintf(r_role, sizeof r_role, "%s/role", base);
    snprintf(r_llm, sizeof r_llm, "%s/llm", base);
    snprintf(r_models, sizeof r_models, "%s/models", base);
    snprintf(r_model,  sizeof r_model,  "%s/model",  base);
    snprintf(r_files,  sizeof r_files,  "%s/files",  base);
    snprintf(r_upload, sizeof r_upload, "%s/upload", base);
    snprintf(r_chat, sizeof r_chat, "%s/chat", base);
    snprintf(r_chat_reset, sizeof r_chat_reset, "%s/chat/reset", base);
    snprintf(r_chat_load, sizeof r_chat_load, "%s/chat/load", base);

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
    } else if (strcmp(method, "GET") == 0 && strcmp(path, r_models) == 0) {
        api_models(w, fd);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, r_model) == 0) {
        api_model(w, fd, body);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, r_files) == 0) {
        api_files(w, fd, query);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, r_files) == 0) {
        api_file_delete(w, fd, body);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, r_upload) == 0) {
        api_upload(w, fd, body, body_len, content_type);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, r_chat) == 0) {
        api_chat_state(w, fd);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, r_chat) == 0) {
        /* Ответ идёт потоком и живёт дольше обычного запроса: пока модель
         * думает, соединение должно держаться, а не отваливаться по таймауту. */
        struct timeval big = { DC_CHAT_TIMEOUT_S, 0 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &big, sizeof big);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &big, sizeof big);
        pthread_mutex_lock(&w->chat_mu);
        w->chat_busy = 1;
        pthread_mutex_unlock(&w->chat_mu);
        api_chat(w, fd, body);
        pthread_mutex_lock(&w->chat_mu);
        w->chat_busy = 0;
        pthread_mutex_unlock(&w->chat_mu);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, r_chat_load) == 0) {
        struct timeval big = { DC_CHAT_TIMEOUT_S, 0 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &big, sizeof big);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &big, sizeof big);
        chat_load(w, fd);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, r_chat_reset) == 0) {
        api_chat_reset(w, fd);
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
    pthread_mutex_init(&w->chat_mu, NULL);
    pthread_mutex_init(&w->up_mu, NULL);
    sb_init(&w->chat_hist);

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
    dc_infer_free(w->chat_infer);
    sb_free(&w->chat_hist);
    pthread_mutex_destroy(&w->mu);
    pthread_mutex_destroy(&w->run_mu);
    pthread_mutex_destroy(&w->chat_mu);
    pthread_mutex_destroy(&w->up_mu);
    free(w);
}
