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

/* Пул рабочих. Рабочий роль не объявляет — он сообщает только мощь, — поэтому
 * распределитель складывает найденных узлов в одну общую секцию «*» и берёт
 * исполнителя оттуда, когда конкретная роль в конфигурации не описана.
 * Секция [role *] в конфигурации тоже читается как пул. */
#define ROLE_POOL     "*"

/* Параметры демона вычислительного узла. Раньше они задавались только
 * CLI-флагами dcd. Теперь их можно определить в конфигурации планировщика —
 * либо глобально в секции [node], либо per-endpoint в секциях [role]. */
typedef struct {
    char *tag;           /* --tag: имя узла */
    char *state_dir;     /* --state: каталог для memory.log */
    int   max_body;      /* --max-body: максимальный размер тела запроса */
    long  pool_bytes;    /* --pool-bytes: размер пула весов модели на узле */
    long  free_mem;      /* --free-mem: память под веса модели */
    int   hello_secs;    /* --hello-interval: интервал broadcast HELLO */
    int   use_tcp;       /* --tcp: слушать TCP порт */
    int   use_dtn;       /* --dtn: DTN поверх TCP */
    char *dtn_queue;     /* --dtn-queue: каталог очереди DTN */
} dc_node_params;

typedef struct {
    char *url;
    long  inflight;      /* runtime */
    long  completed;     /* runtime */
    long  failed;        /* runtime */
    dc_capability cap;   /* что узел умеет; заполняется по PONG/HELLO */
    dc_node_params params; /* per-endpoint параметры (переопределяют глобальные) */
} dc_endpoint;

typedef struct {
    char        *name;
    dc_endpoint  eps[DC_MAX_EPS];
    int          nep;
} dc_role;

typedef struct {
    /* [llm] — модель, которая строит план. Она не внешняя: файл GGUF лежит на
     * диске, а веса раскладываются по узлам роли shard. Поэтому здесь нет ни
     * адреса сервера, ни ключа, ни стиля запроса. */
    char *model_file;    /* путь к GGUF; пусто — план строит эвристика */
    int   n_ctx;         /* токенов контекста; 0 — 512, потолок 4096 */
    int   max_new;       /* сколько токенов модель может нагенерировать */
    char *payload_mode;  /* expr | free */

    /* [chat] — веб-чат. Модель та же и разложена так же, но системный
     * промпт другой: здесь она отвечает человеку, а не строит план. */
    char *chat_system;   /* свой текст вместо встроенного; пусто — встроенный */
    int   chat_history;  /* сколько последних сообщений класть в контекст */

    /* [role <name>] */
    dc_role roles[DC_MAX_ROLES];
    int     nroles;

    /* [scheduler] */
    int   concurrency;
    int   retries;
    long  timeout_ms;
    long  backoff_ms;
    char *outdir;

    /* [node] — глобальные параметры демона для всех узлов */
    dc_node_params node_params;
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

/* Эффективные параметры узла: per-endpoint переопределяют глобальные [node].
 * Возвращает указатель на статический буфер (не сохранять). */
dc_node_params *conf_effective_params(const dc_config *cfg, const dc_endpoint *ep);

/* Генерирует команду запуска dcd для данного узла. Роль в команду не идёт:
 * рабочий её не объявляет. Возвращает malloc'нутую строку. */
char *conf_launch_cmd(const dc_config *cfg, const char *role, const dc_endpoint *ep);

#endif /* DC_CONF_H */
