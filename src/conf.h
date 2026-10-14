#ifndef DC_CONF_H
#define DC_CONF_H

#include "udp.h"

#include <stddef.h>

/* Версия печатается в --version и уходит в User-Agent. */
#define DC_VERSION "1.1"

#define DC_MAX_ROLES 16
#define DC_MAX_EPS   64

#define ROLE_COMPUTE  "compute"
#define ROLE_VERIFY   "verify"
#define ROLE_MEMORIZE "memorize"
#define ROLE_SHARD    "shard"   /* часть модели: держит свои слои весов */

typedef struct {
    char *url;
    long  inflight;   /* runtime */
    long  completed;  /* runtime */
    long  failed;     /* runtime */
    dc_capability cap; /* что узел умеет; заполняется по PONG/HELLO */
} dc_endpoint;

typedef struct {
    char        *name;
    dc_endpoint  eps[DC_MAX_EPS];
    int          nep;
} dc_role;

typedef struct {
    /* [llm] */
    char  *llm_url;
    char  *llm_model;
    char  *llm_key;
    char  *llm_style;        /* openai | anthropic */
    char  *llm_auth_header;  /* e.g. x-api-key */
    char  *llm_extra_header; /* e.g. anthropic-version: 2023-06-01 */
    char  *llm_json_mode;    /* auto | on | off */
    /* как обращаться к LLM: auto | none | bearer.
     * auto — ключ есть, значит шлём; ключа нет, значит сервер локальный или
     * свой в сети, и авторизация ему не нужна. Именно так работают
     * llama.cpp server, Ollama и LM Studio, запущенные без ключа. */
    char  *llm_auth;
    long   llm_timeout_ms;
    double llm_temp;
    char  *payload_mode;     /* expr | free */

    /* [role <name>] */
    dc_role roles[DC_MAX_ROLES];
    int     nroles;

    /* [scheduler] */
    int   concurrency;
    int   retries;
    long  timeout_ms;
    long  backoff_ms;
    char *outdir;
} dc_config;

dc_config *conf_new(void);
void       conf_free(dc_config *cfg);
dc_config *conf_load(const char *path, char *err, size_t errlen);
int        conf_save_example(const char *path);

/* Глубокая копия конфигурации. Нужна, чтобы выполняющийся в фоне запуск
 * работал со своим снимком и не спорил с правками ролей из интерфейса. */
dc_config *conf_clone(const dc_config *src);

dc_role *conf_role(dc_config *cfg, const char *name, int create);
int      conf_role_add_ep(dc_role *r, const char *url);
int      conf_has_daemons(const dc_config *cfg);

#endif /* DC_CONF_H */
