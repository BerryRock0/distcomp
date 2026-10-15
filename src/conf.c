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
    /* Модели по умолчанию нет: без неё план строится эвристикой, и об этом
     * честно говорит llm check, а не молчаливый откат. */
    cfg->model_file   = xstrdup("");
    cfg->n_ctx        = 0;      /* 0 — движок подставит разумное значение */
    cfg->max_new      = 512;
    cfg->payload_mode = xstrdup("expr");
    cfg->concurrency  = 4;
    cfg->retries      = 2;
    cfg->timeout_ms   = 60000;
    cfg->backoff_ms   = 500;
    cfg->outdir       = xstrdup("runs");
    return cfg;
}

void conf_free(dc_config *cfg)
{
    if (!cfg) return;
    free(cfg->model_file);
    free(cfg->payload_mode);
    free(cfg->outdir);
    for (int i = 0; i < cfg->nroles; i++) {
        free(cfg->roles[i].name);
        for (int j = 0; j < cfg->roles[i].nep; j++) free(cfg->roles[i].eps[j].url);
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
    set_str(&d->outdir, src->outdir);
    d->n_ctx        = src->n_ctx;
    d->max_new      = src->max_new;
    d->concurrency  = src->concurrency;
    d->retries      = src->retries;
    d->timeout_ms   = src->timeout_ms;
    d->backoff_ms   = src->backoff_ms;

    d->nroles = 0;
    for (int i = 0; i < src->nroles && d->nroles < DC_MAX_ROLES; i++) {
        dc_role *r = conf_role(d, src->roles[i].name, 1);
        if (!r) break;
        for (int j = 0; j < src->roles[i].nep; j++) {
            conf_role_add_ep(r, src->roles[i].eps[j].url);
            r->eps[r->nep - 1].completed = src->roles[i].eps[j].completed;
            r->eps[r->nep - 1].failed = src->roles[i].eps[j].failed;
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
                    if (conf_role_add_ep(r, value) != 0) {
                        snprintf(err, errlen, "%s:%d: слишком много узлов в роли %s",
                                 file, lineno, r->name);
                        free(value);
                        goto fail;
                    }
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
        } else if (str_ieq(section, "scheduler")) {
            if (str_ieq(key, "concurrency"))     cfg->concurrency = (int)strtol(value, NULL, 10);
            else if (str_ieq(key, "retries"))     cfg->retries = (int)strtol(value, NULL, 10);
            else if (str_ieq(key, "timeout_ms"))  cfg->timeout_ms = strtol(value, NULL, 10);
            else if (str_ieq(key, "backoff_ms"))  cfg->backoff_ms = strtol(value, NULL, 10);
            else if (str_ieq(key, "outdir"))      set_str(&cfg->outdir, value);
            else log_debug("неизвестный ключ scheduler.%s", key);
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
        "[role %s]\n"
        "daemon = 127.0.0.1:9101       # вычисления (host:port)\n"
        "daemon = 127.0.0.1:9102       # второй вычислитель (балансировка)\n"
        "daemon = tcp://127.0.0.1:9111 # тот же узел через TCP, если UDP не годится\n"
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
        "[scheduler]\n"
        "concurrency = 4\n"
        "retries     = 2\n"
        "timeout_ms  = 60000\n"
        "backoff_ms  = 500\n"
        "outdir      = runs\n",
        ROLE_SHARD, ROLE_COMPUTE, ROLE_VERIFY, ROLE_MEMORIZE, ROLE_SHARD);
    fclose(f);
    return 0;
}
