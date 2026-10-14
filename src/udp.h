#ifndef DC_UDP_H
#define DC_UDP_H

#include <stddef.h>
#include <stdint.h>

/*
 * Транспорт планировщик <-> узлы поверх UDP.
 *
 * Формат дейтаграммы (big endian):
 *
 *   смещение  размер  поле
 *      0        2     magic = 0x4443 ('D','C')
 *      2        1     версия протокола
 *      3        1     тип сообщения
 *      4        2     флаги (бит 0 — тело в JSON)
 *      6        4     длина тела
 *     10        4     CRC32 тела
 *     14        N     тело (JSON, UTF-8)
 *
 * Типы сообщений: REQUEST, RESPONSE, PING, PONG.
 * Ответ всегда приходит на тот же адрес и порт, с которого ушёл запрос,
 * поэтому источник проверяется — чужая «зависшая» дейтаграмма не будет
 * принята за ответ.
 */

#define DC_UDP_MAGIC       0x4443u
#define DC_UDP_VERSION     2
#define DC_UDP_HEADER_LEN  14
#ifndef DC_UDP_MAX_DGRAM           /* переопределяется сборкой lite */
#define DC_UDP_MAX_DGRAM   60000   /* с запасом под MTU и служебные заголовки */
#endif
#define DC_UDP_FLAG_JSON   0x0001

/* Порт, на котором узлы рассылают HELLO при старте и периодически.
 * Планировщик слушает его командой `dcctl discover`, чтобы узлы находились
 * без ручной правки конфигурации. */
#define DC_DISCOVERY_PORT  49152

enum {
    DC_MSG_REQUEST  = 1,
    DC_MSG_RESPONSE = 2,
    DC_MSG_PING     = 3,
    DC_MSG_PONG     = 4,
    DC_MSG_HELLO    = 5,   /* узел -> сеть: «я такой-то, вот мои роли и лимит» */
    DC_MSG_WELCOME  = 6    /* сеть -> узел: «меня услышали» */
};

/* Что узел умеет и чего может. Планировщик читает это из ответа и не
 * отправляет узлу шаг тяжелее max_body. free_mem — сколько байт весов
 * модели узел готов держать: по этому и раскладывается шардированная
 * модель между устройствами с разным объёмом памяти. */
typedef struct {
    int   max_body;    /* максимальный размер тела, который узел примет, байт */
    long  free_mem;    /* сколько байт под веса узел отдаёт под модель */
    char  roles[96];   /* список ролей через запятую, для отладки и discover */
    char  arch[48];    /* краткое имя платформы */
    char  node[32];    /* имя узла */
    char  shard[96];   /* описание shard-узла: «слои 2-5, вход 16, 1.2 МБ» */
    int   known;       /* capability уже получены */
} dc_capability;

typedef struct {
    unsigned char       type;
    unsigned short      flags;
    const unsigned char *body;
    size_t              body_len;
} dc_msg;

uint32_t udp_crc32(const void *data, size_t len);

/* Упаковывает JSON-тело в дейтаграмму. Возвращает malloc'нутый буфер. */
int udp_pack(unsigned char type, const char *json, unsigned char **out, size_t *out_len);

/* Как udp_pack, но для широковещательного HELLO. port — порт службы узла,
 * а не порт отправителя; free_mem и shard идут в объявление, чтобы
 * планировщик видел память устройства ещё до того, как начнёт раскладывать
 * по нему модель. */
int udp_pack_hello(int max_body, const char *roles, const char *arch, const char *node,
                   int port, long free_mem, const char *shard, unsigned char **out,
                   size_t *out_len);

/* Разбирает дейтаграмму. body указывает внутрь buf — копировать не нужно. */
int udp_unpack(const unsigned char *buf, size_t len, dc_msg *msg, char *err, size_t errlen);

/* Сокет для клиента (пока не привязан) и для сервера (привязан к host:port). */
int udp_socket_client(char *err, size_t errlen);
int udp_socket_bound(const char *host, int port, char *err, size_t errlen);

/* То же, но с заданными размерами буферов приёма и отправки — для сокета
 * объявлений на микроконтроллере мегабайтный буфер недопустим. */
int udp_socket_bound_ex(const char *host, int port, int rcvbuf, int sndbuf, char *err,
                        size_t errlen);

/* Один запрос-ответ. Возвращает 0 при получении корректного RESPONSE/PONG,
 * тело ответа (JSON) кладётся в *out_body — его нужно free(). */
int udp_request(int sock, const char *host, int port, unsigned char type,
                const char *json, long timeout_ms, char **out_body, char *err, size_t errlen);

/* Как udp_request, но payload — произвольные данные, а не JSON. Команды
 * узлу LLM (wt|…, gen|…) не являются JSON, и запихивать их в JSON-строку
 * значило бы экранировать base64 и удлинять каждую дейтаграмму. */
int udp_request_raw(int sock, const char *host, int port, const char *payload, size_t len,
                    long timeout_ms, char **out_body, char *err, size_t errlen);

/* Разбирает "http://host:port" или "host:port" в адрес для UDP. */
int udp_parse_endpoint(const char *url, char *host, size_t hostlen, int *port, char *err,
                       size_t errlen);

#endif /* DC_UDP_H */
