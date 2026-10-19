#include "conf.h"
#include "util.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void set_str(char **slot, const char *val)
{
    free(*slot);
    *slot = val ? xstrdup(val) : NULL;
}

/* Значение вида "$VAR" или "Bearer $VAR" подтягивается из окружения. */
static char *expand_value(const char *val)
{
    if (!val) return NULL;
    const char *dollar = strchr(val, '$');
    if (!dollar) return xstrdup(val);

    strbuf b;
    sb_init(&b);
    sb_addn(&b, val, (size_t)(dollar - val));

    const char *p = dollar + 1;
    while (*p && (isalnum((unsigned char)*p) || *p == '_')) sb_addc(&b, *p++);
    char var[256];
    size_t vl = (size_t)(p - (dollar + 1));
    if (vl == 0 || vl >= sizeof var) vl = sizeof var - 1;
    memcpy(var, dollar + 1, vl);
    var[vl] = '\0';

    const char *env = getenv(var);
    if (env) sb_add(&b, env);
    else log_debug("переменная окружения %s не задана", var);

    if (*p) sb_add(&b, p);
    char *out = b.p ? b.p : xstrdup("");
    return out;
}

dc_config *conf_new(void)
{
    dc_config *cfg = xcalloc(1, sizeof *cfg);
    /* Модели по умолчанию нет: без неё план строется эвристикой, и об этом
     * честно говорит llm check, а не молчаливый откат. */
    cfg->model_file   = xstrdup("");
    cfg->n_ctx        = 0;      /* 0 — движок подставит разумное значение */
    cfg->max_new      = 512;
    cfg->payload_mode = xstrdup("expr");
    /* чат: пустой system — встроенный текст, 8 сообщений истории достаточно
     * для обычного разговора и не съедает контекст целиком */
    cfg->chat_system  = xstrdup("");
    cfg->chat_history = 8;
    cfg->concurrency  = 4;
    cfg->retries      = 2;
    cfg->timeout_ms   = 60000;
    cfg->backoff_ms   = 500;
    cfg->outdir       = xstrdup("runs");
    cfg->upload_dir   = xstrdup("uploads");
    /* Параметры демона по умолчанию (как в dcd) */
    cfg->node_params.hello_secs = 10;  /* DC_HELLO_SECS */
    /* 0 — «все свободные ядра»: тяжёлой ноде незачем беречь их впустую */
    cfg->node_params.threads    = 0;
    return cfg;
}

void conf_free(dc_config *cfg)
{
    if (!cfg) return;
    free(cfg->model_file);
    free(cfg->payload_mode);
    free(cfg->chat_system);
    free(cfg->outdir);
    free(cfg->upload_dir);
    free(cfg->node_params.tag);
    free(cfg->node_params.state_dir);
    free(cfg->node_params.dtn_queue);
    for (int i = 0; i < cfg->nroles; i++) {
        free(cfg->roles[i].name);
        for (int j = 0; j < cfg->roles[i].nep; j++) {
            free(cfg->roles[i].eps[j].url);
            free(cfg->roles[i].eps[j].params.tag);
            free(cfg->roles[i].eps[j].params.state_dir);
            free(cfg->roles[i].eps[j].params.dtn_queue);
        }
    }
    free(cfg);
}

dc_role *conf_role(dc_config *cfg, const char *name, int create)
{
    for (int i = 0; i < cfg->nroles; i++)
        if (str_ieq(cfg->roles[i].name, name)) return &cfg->roles[i];
    if (!create) return NULL;
    if (cfg->nroles >= DC_MAX_ROLES) return NULL;
    dc_role *r = &cfg->roles[cfg->nroles++];
    memset(r, 0, sizeof *r);
    r->name = xstrdup(name);
    return r;
}

int conf_role_add_ep(dc_role *r, const char *url)
{
    if (!r || r->nep >= DC_MAX_EPS) return -1;
    /* одинаковые адреса не дублируем */
    for (int i = 0; i < r->nep; i++)
        if (strcmp(r->eps[i].url, url) == 0) return 0;
    r->eps[r->nep].url = xstrdup(url);
    r->eps[r->nep].inflight = 0;
    /* -1 — «про этот узел не сказали». Отличать надо: ноль у threads означает
     * «считай на всех ядрах» и вовсе не то же, что «параметра нет». */
    r->eps[r->nep].params.threads = -1;
    r->nep++;
    return 0;
}

int conf_has_daemons(const dc_config *cfg)
{
    for (int i = 0; i < cfg->nroles; i++)
        if (cfg->roles[i].nep > 0) return 1;
    return 0;
}

dc_config *conf_clone(const dc_config *src)
{
    if (!src) return NULL;
    dc_config *d = conf_new();
    set_str(&d->model_file, src->model_file);
    set_str(&d->payload_mode, src->payload_mode);
    set_str(&d->chat_system, src->chat_system);
    set_str(&d->outdir, src->outdir);
    set_str(&d->upload_dir, src->upload_dir);
    d->n_ctx        = src->n_ctx;
    d->max_new      = src->max_new;
    d->chat_history = src->chat_history;
    d->concurrency  = src->concurrency;
    d->retries      = src->retries;
    d->timeout_ms   = src->timeout_ms;
    d->backoff_ms   = src->backoff_ms;

    /* Копируем глобальные параметры демона */
    set_str(&d->node_params.tag, src->node_params.tag);
    set_str(&d->node_params.state_dir, src->node_params.state_dir);
    set_str(&d->node_params.dtn_queue, src->node_params.dtn_queue);
    d->node_params.max_body   = src->node_params.max_body;
    d->node_params.pool_bytes = src->node_params.pool_bytes;
    d->node_params.free_mem   = src->node_params.free_mem;
    d->node_params.hello_secs = src->node_params.hello_secs;
    d->node_params.threads    = src->node_params.threads;
    d->node_params.use_tcp    = src->node_params.use_tcp;
    d->node_params.use_dtn    = src->node_params.use_dtn;

    d->nroles = 0;
    for (int i = 0; i < src->nroles && d->nroles < DC_MAX_ROLES; i++) {
        dc_role *r = conf_role(d, src->roles[i].name, 1);
        if (!r) break;
        for (int j = 0; j < src->roles[i].nep; j++) {
            conf_role_add_ep(r, src->roles[i].eps[j].url);
            r->eps[r->nep - 1].completed = src->roles[i].eps[j].completed;
            r->eps[r->nep - 1].failed = src->roles[i].eps[j].failed;
            /* Копируем per-endpoint параметры */
            set_str(&r->eps[r->nep - 1].params.tag, src->roles[i].eps[j].params.tag);
            set_str(&r->eps[r->nep - 1].params.state_dir, src->roles[i].eps[j].params.state_dir);
            set_str(&r->eps[r->nep - 1].params.dtn_queue, src->roles[i].eps[j].params.dtn_queue);
            r->eps[r->nep - 1].params.max_body   = src->roles[i].eps[j].params.max_body;
            r->eps[r->nep - 1].params.pool_bytes = src->roles[i].eps[j].params.pool_bytes;
            r->eps[r->nep - 1].params.free_mem   = src->roles[i].eps[j].params.free_mem;
            r->eps[r->nep - 1].params.hello_secs = src->roles[i].eps[j].params.hello_secs;
            r->eps[r->nep - 1].params.threads    = src->roles[i].eps[j].params.threads;
            r->eps[r->nep - 1].params.use_tcp    = src->roles[i].eps[j].params.use_tcp;
            r->eps[r->nep - 1].params.use_dtn    = src->roles[i].eps[j].params.use_dtn;
        }
    }
    return d;
}

dc_config *conf_load(const char *path, char *err, size_t errlen)
{
    char        *used = NULL;
    const char *file = path;

    if (!file) {
        if (access("distcomp.conf", R_OK) == 0) {
            file = "distcomp.conf";
        } else {
            const char *home = getenv("HOME");
            char        buf[1024];
            if (home && snprintf(buf, sizeof buf, "%s/.distcomp.conf", home) > 0 &&
                access(buf, R_OK) == 0)
                file = xstrdup(buf);
            else
                file = NULL;
        }
    }
    if (!file) {
        snprintf(err, errlen,
                 "файл конфигурации не найден (ищем ./distcomp.conf или ~/.distcomp.conf); "
                 "укажите -c ПУТЬ");
        return NULL;
    }
    used = xstrdup(file);

    FILE *f = fopen(file, "r");
    if (!f) {
        snprintf(err, errlen, "не удалось открыть %s", file);
        free(used);
        return NULL;
    }

    dc_config *cfg = conf_new();
    char        line[4096];
    char        section[128] = "scheduler";
    int         lineno = 0;

    while (fgets(line, sizeof line, f)) {
        lineno++;
        char *s = str_trim(line);
        if (!*s || *s == '#' || *s == ';') continue;

        if (*s == '[') {
            char *end = strchr(s, ']');
            if (!end) {
                snprintf(err, errlen, "%s:%d: не закрыта секция", file, lineno);
                goto fail;
            }
            *end = '\0';
            char *name = str_trim(s + 1);
            snprintf(section, sizeof section, "%s", name);
            continue;
        }

        char *eq = strchr(s, '=');
        if (!eq) continue;
        *eq = '\0';
        char *key = str_trim(s);
        char *val = str_trim(eq + 1);

        /* Комментарий в конце значения. Знак «#» считается началом комментария
         * только на месте или после пробела: так «# внутри пароля» остаётся
         * паролем, а «daemon = 1.2.3.4:9101 # вычисления» — адресом. */
        for (char *c = val; *c; c++) {
            if (*c == '#' && (c == val || c[-1] == ' ' || c[-1] == '\t')) {
                *c = '\0';
                val = str_trim(val);
                break;
            }
        }
        if (!*key) continue;

        char *value = expand_value(val);

        /* секция "role compute" или "role:compute" */
        if (strncasecmp(section, "role", 4) == 0 &&
            (section[4] == '\0' || section[4] == ':' || section[4] == ' ')) {
            const char *rname = str_trim((char *)section + 4);
            if (*rname == ':' || *rname == '=') rname = str_trim((char *)rname + 1);
            if (*rname) {
                dc_role *r = conf_role(cfg, rname, 1);
                if (!r) {
                    snprintf(err, errlen, "%s:%d: слишком много ролей", file, lineno);
                    free(value);
                    goto fail;
                }
                if (str_ieq(key, "daemon") || str_ieq(key, "url") || str_ieq(key, "endpoint")) {
                    /* Формат: daemon = host:port [key=val ...]
                     * Дополнительные параметры через пробел после адреса */
                    char *val_copy = xstrdup(value);
                    char *sp = strpbrk(val_copy, " \t");
                    char *ep_params = NULL;
                    if (sp) {
                        *sp = '\0';
                        ep_params = str_trim(sp + 1);
                    }
                    if (conf_role_add_ep(r, val_copy) != 0) {
                        snprintf(err, errlen, "%s:%d: слишком много узлов в роли %s",
                                 file, lineno, r->name);
                        free(val_copy);
                        free(value);
                        goto fail;
                    }
                    /* Парсим per-endpoint параметры */
                    if (ep_params && *ep_params) {
                        dc_endpoint *ep = &r->eps[r->nep - 1];
                        char *tok = strtok(ep_params, " \t");
                        while (tok) {
                            char *eq = strchr(tok, '=');
                            if (eq) {
                                *eq = '\0';
                                char *k = str_trim(tok);
                                char *v = str_trim(eq + 1);
                                if (str_ieq(k, "tag")) set_str(&ep->params.tag, v);
                                else if (str_ieq(k, "state")) set_str(&ep->params.state_dir, v);
                                else if (str_ieq(k, "state_dir")) set_str(&ep->params.state_dir, v);
                                else if (str_ieq(k, "max_body") || str_ieq(k, "max-body")) ep->params.max_body = (int)strtol(v, NULL, 10);
                                else if (str_ieq(k, "pool_bytes") || str_ieq(k, "pool-bytes")) ep->params.pool_bytes = strtol(v, NULL, 10);
                                else if (str_ieq(k, "free_mem") || str_ieq(k, "free-mem")) ep->params.free_mem = strtol(v, NULL, 10);
                                else if (str_ieq(k, "hello_interval") || str_ieq(k, "hello-interval")) ep->params.hello_secs = (int)strtol(v, NULL, 10);
                                else if (str_ieq(k, "threads")) ep->params.threads = (int)strtol(v, NULL, 10);
                                else if (str_ieq(k, "tcp")) ep->params.use_tcp = atoi(v) != 0;
                                else if (str_ieq(k, "dtn")) ep->params.use_dtn = atoi(v) != 0;
                                else if (str_ieq(k, "dtn_queue") || str_ieq(k, "dtn-queue")) set_str(&ep->params.dtn_queue, v);
                                else log_debug("неизвестный параметр узла %s=%s", k, v);
                            }
                            tok = strtok(NULL, " \t");
                        }
                    }
                    free(val_copy);
                    log_debug("роль %s -> %s", r->name, value);
                }
                free(value);
                continue;
            }
        }

        if (str_ieq(section, "llm")) {
            if (str_ieq(key, "model_file"))       set_str(&cfg->model_file, value);
            else if (str_ieq(key, "model"))        set_str(&cfg->model_file, value);
            else if (str_ieq(key, "n_ctx"))        cfg->n_ctx = (int)strtol(value, NULL, 10);
            else if (str_ieq(key, "max_new"))      cfg->max_new = (int)strtol(value, NULL, 10);
            else if (str_ieq(key, "payload_mode")) set_str(&cfg->payload_mode, value);
            else log_debug("неизвестный ключ llm.%s", key);
        } else if (str_ieq(section, "chat")) {
            if (str_ieq(key, "system"))        set_str(&cfg->chat_system, value);
            else if (str_ieq(key, "history"))  cfg->chat_history = (int)strtol(value, NULL, 10);
            else log_debug("неизвестный ключ chat.%s", key);
        } else if (str_ieq(section, "scheduler")) {
            if (str_ieq(key, "concurrency"))     cfg->concurrency = (int)strtol(value, NULL, 10);
            else if (str_ieq(key, "retries"))     cfg->retries = (int)strtol(value, NULL, 10);
            else if (str_ieq(key, "timeout_ms"))  cfg->timeout_ms = strtol(value, NULL, 10);
            else if (str_ieq(key, "backoff_ms"))  cfg->backoff_ms = strtol(value, NULL, 10);
            else if (str_ieq(key, "outdir"))      set_str(&cfg->outdir, value);
            else if (str_ieq(key, "upload_dir"))   set_str(&cfg->upload_dir, value);
            else log_debug("неизвестный ключ scheduler.%s", key);
        } else if (str_ieq(section, "node")) {
            /* Глобальные параметры демона для всех узлов */
            if (str_ieq(key, "tag"))           set_str(&cfg->node_params.tag, value);
            else if (str_ieq(key, "state"))     set_str(&cfg->node_params.state_dir, value);
            else if (str_ieq(key, "state_dir")) set_str(&cfg->node_params.state_dir, value);
            else if (str_ieq(key, "max_body"))  cfg->node_params.max_body = (int)strtol(value, NULL, 10);
            else if (str_ieq(key, "max-body")) cfg->node_params.max_body = (int)strtol(value, NULL, 10);
            else if (str_ieq(key, "pool_bytes")) cfg->node_params.pool_bytes = strtol(value, NULL, 10);
            else if (str_ieq(key, "pool-bytes")) cfg->node_params.pool_bytes = strtol(value, NULL, 10);
            else if (str_ieq(key, "free_mem"))  cfg->node_params.free_mem = strtol(value, NULL, 10);
            else if (str_ieq(key, "free-mem"))  cfg->node_params.free_mem = strtol(value, NULL, 10);
            else if (str_ieq(key, "hello_interval")) cfg->node_params.hello_secs = (int)strtol(value, NULL, 10);
            else if (str_ieq(key, "hello-interval")) cfg->node_params.hello_secs = (int)strtol(value, NULL, 10);
            else if (str_ieq(key, "threads")) cfg->node_params.threads = (int)strtol(value, NULL, 10);
            else if (str_ieq(key, "tcp"))       cfg->node_params.use_tcp = atoi(value) != 0;
            else if (str_ieq(key, "dtn"))       cfg->node_params.use_dtn = atoi(value) != 0;
            else if (str_ieq(key, "dtn_queue")) set_str(&cfg->node_params.dtn_queue, value);
            else if (str_ieq(key, "dtn-queue")) set_str(&cfg->node_params.dtn_queue, value);
            else log_debug("неизвестный ключ node.%s", key);
        }
        free(value);
    }

    fclose(f);
    free(used);
    log_debug("конфигурация загружена из %s", file);
    return cfg;

fail:
    fclose(f);
    free(used);
    conf_free(cfg);
    return NULL;
}

int conf_save_example(const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f,
        "# distcomp — конфигурация планировщика распределённых вычислений\n"
        "# Пользователь сам решает, какой узел какую роль обслуживает.\n"
        "\n"
        "[llm]\n"
        "# Модель, которая разлагает цель на шаги. Это не внешний сервер: файл GGUF\n"
        "# читается локально, а его веса раскладываются по узлам роли %s.\n"
        "# Пока model_file пуст, план строится эвристикой.\n"
        "model_file  = model.gguf\n"
        "n_ctx       = 1024            # токенов контекста; больше — узлам нужно\n"
        "                             # больше памяти под KV-кэш (потолок 4096)\n"
        "max_new     = 512             # сколько токенов модель может нагенерировать\n"
        "payload_mode = expr           # expr — payload это арифметика; free — произвольный текст\n"
        "\n"
        "[chat]\n"
        "# Веб-чат. Модель и разложение те же, что в [llm], но системный промпт\n"
        "# другой: здесь модель отвечает человеку. Пустая строка — встроенный\n"
        "# текст, своей строкой можно задать свой.\n"
        "system  =\n"
        "history = 8                    # сколько последних сообщений класть в контекст\n"
        "\n"
        "[node]\n"
        "# Глобальные параметры демона для ВСЕХ узлов. Можно переопределить\n"
        "# per-endpoint в секциях [role]: daemon = host:port tag=mytag max_body=59986\n"
        "# tag           = node            # имя узла в отчётах\n"
        "# state         = state           # каталог для memory.log\n"
        "# max_body      = 59986           # максимальный размер тела запроса (байт)\n"
        "# free_mem      = 1048576         # память под веса модели (байт)\n"
        "# pool_bytes    = 268435456       # размер пула весов, который узел выделяет\n"
        "#                              # при старте (--pool-bytes). Это потолок:\n"
        "#                              # больше free_mem он быть не может.\n"
        "#                              # Уже работающий узел пул не меняет —\n"
        "#                              # применяется при перезапуске.\n"
        "# hello_interval = 10              # интервал broadcast HELLO (сек, 0 = выкл)\n"
        "# threads       = 0               # сколько ядер отдать под счёт:\n"
        "#                              # 0 (по умолчанию) — все свободные.\n"
        "#                              # Тяжёлой ноде незачем беречь их\n"
        "#                              # впустую: на 16 ядрах считает\n"
        "#                              # вшестеро быстрее одного. 1 — как\n"
        "#                              # было, если ядер не надо делить.\n"
        "# tcp           = 0               # слушать TCP порт\n"
        "# dtn           = 0               # DTN поверх TCP\n"
        "# dtn_queue     = state/dtn       # каталог очереди DTN\n"
        "\n"
        "[role %s]\n"
        "daemon = 127.0.0.1:9101       # вычисления (host:port)\n"
        "daemon = 127.0.0.1:9102       # второй вычислитель (балансировка)\n"
        "daemon = tcp://127.0.0.1:9111 # тот же узел через TCP, если UDP не годится\n"
        "# per-endpoint параметры: daemon = 127.0.0.1:9101 tag=compute-1 max_body=59986\n"
        "\n"
        "[role %s]\n"
        "daemon = 127.0.0.1:9103       # проверка\n"
        "\n"
        "[role %s]\n"
        "daemon = 127.0.0.1:9104       # запоминание\n"
        "\n"
        "[role %s]\n"
        "# Устройства, на которых живут веса модели. Нужно минимум три:\n"
        "# вход (эмбеддинг), слои и выход (голова) обязаны быть на разных узлах.\n"
        "daemon = 127.0.0.1:9105\n"
        "daemon = 127.0.0.1:9106\n"
        "daemon = 127.0.0.1:9107\n"
        "\n"
        "\n"
        "Секции [role ...] необязательны: если их нет, распределитель сам\n"
        "найдёт рабочих в сети и назначит роль каждому на каждой задаче.\n"
        "Секция нужна, чтобы оставить конкретный узел за конкретной ролью.\n"
        "\n"
        "[scheduler]\n"
        "concurrency = 4\n"
        "retries     = 2\n"
        "timeout_ms  = 60000\n"
        "backoff_ms  = 500\n"
        "outdir      = runs\n"
        "upload_dir  = uploads            # сюда панель складывает загруженные файлы\n",
        ROLE_SHARD, ROLE_COMPUTE, ROLE_VERIFY, ROLE_MEMORIZE, ROLE_SHARD);
    fclose(f);
    return 0;
}

/* Эффективные параметры узла: per-endpoint переопределяют глобальные [node].
 * Возвращает указатель на статический буфер (не сохранять). */
dc_node_params *conf_effective_params(const dc_config *cfg, const dc_endpoint *ep)
{
    static dc_node_params eff;
    const dc_node_params *g = &cfg->node_params;
    const dc_node_params *p = ep ? &ep->params : NULL;

    eff.tag        = (p && p->tag)        ? p->tag        : g->tag;
    eff.state_dir  = (p && p->state_dir)  ? p->state_dir  : g->state_dir;
    eff.dtn_queue  = (p && p->dtn_queue)  ? p->dtn_queue  : g->dtn_queue;
    eff.max_body   = (p && p->max_body > 0)   ? p->max_body   : g->max_body;
    eff.pool_bytes = (p && p->pool_bytes > 0) ? p->pool_bytes : g->pool_bytes;
    eff.free_mem   = (p && p->free_mem > 0)   ? p->free_mem   : g->free_mem;
    eff.hello_secs = (p && p->hello_secs > 0) ? p->hello_secs : g->hello_secs;
    /* Ноль у threads — осмысленное значение («все ядра»), поэтому здесь
     * знаком именно -1: не задан узел — берём общее из [node]. */
    eff.threads    = (p && p->threads >= 0) ? p->threads : g->threads;
    eff.use_tcp    = (p && p->use_tcp)    ? p->use_tcp    : g->use_tcp;
    eff.use_dtn    = (p && p->use_dtn)    ? p->use_dtn    : g->use_dtn;

    return &eff;
}

/* Генерирует команду запуска dcd для данного узла. Возвращает malloc'нутую строку. */
char *conf_launch_cmd(const dc_config *cfg, const char *role, const dc_endpoint *ep)
{
    if (!ep) return NULL;
    const dc_node_params *p = conf_effective_params(cfg, ep);

    strbuf b;
    sb_init(&b);
    sb_add(&b, "dcd");

    /* Роль в команде запуска больше не передаётся: рабочий её не объявляет
     * и не проверяет, а назначает распределитель в теле каждой задачи.
     * Раньше здесь стояло жёсткое «--role compute», и `dcctl launch` для роли
     * shard предлагал запустить вычислитель. */
    (void)role;

    /* Порт из URL — поддерживаем udp://, tcp://, dtn:// и host:port */
    char host[256];
    int  port = 0;
    char eperr[128] = {0};
    if (udp_parse_endpoint(ep->url, host, sizeof host, &port, eperr, sizeof eperr) == 0 && port > 0) {
        char portstr[16];
        snprintf(portstr, sizeof portstr, "%d", port);
        sb_add(&b, " --port ");
        sb_add(&b, portstr);
    } else {
        /* Fallback: пытаемся извлечь порт из строки вручную */
        const char *colon = strrchr(ep->url, ':');
        if (colon && colon[1] >= '0' && colon[1] <= '9') {
            sb_add(&b, " --port ");
            sb_add(&b, colon + 1);
        }
    }

    if (p->tag) {
        sb_add(&b, " --tag ");
        sb_add(&b, p->tag);
    }
    if (p->state_dir) {
        sb_add(&b, " --state ");
        sb_add(&b, p->state_dir);
    }
    if (p->max_body > 0) {
        char buf[16];
        snprintf(buf, sizeof buf, "%d", p->max_body);
        sb_add(&b, " --max-body ");
        sb_add(&b, buf);
    }
    if (p->pool_bytes > 0) {
        char buf[32];
        snprintf(buf, sizeof buf, "%ld", p->pool_bytes);
        sb_add(&b, " --pool-bytes ");
        sb_add(&b, buf);
    }
    if (p->free_mem > 0) {
        char buf[32];
        snprintf(buf, sizeof buf, "%ld", p->free_mem);
        sb_add(&b, " --free-mem ");
        sb_add(&b, buf);
    }
    /* Всегда пишем: ноль означает «все ядра», и читать команду должно быть
     * видно, что запускается именно тяжёлый, а не облегчённый узел. */
    {
        char tbuf[16];
        snprintf(tbuf, sizeof tbuf, "%d", p->threads);
        sb_add(&b, " --threads ");
        sb_add(&b, tbuf);
    }
    if (p->hello_secs > 0) {
        char buf[16];
        snprintf(buf, sizeof buf, "%d", p->hello_secs);
        sb_add(&b, " --hello-interval ");
        sb_add(&b, buf);
    }
    if (p->use_tcp) {
        sb_add(&b, " --tcp");
    }
    if (p->use_dtn) {
        sb_add(&b, " --dtn");
    }
    if (p->dtn_queue) {
        sb_add(&b, " --dtn-queue ");
        sb_add(&b, p->dtn_queue);
    }

    return b.p ? b.p : xstrdup("dcd");
}
