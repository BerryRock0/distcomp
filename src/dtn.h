#ifndef DC_DTN_H
#define DC_DTN_H

#include <stddef.h>
#include <stdint.h>

/*
 * DTN (Delay/Disruption-Tolerant Networking) поверх TCP.
 *
 * В отличие от «просто TCP», канал может быть плохим: большая задержка,
 * разрывы, потери. Поэтому каждое сообщение — это bundle с номером:
 * перед отправкой он кладётся в очередь на диск, снимается оттуда только
 * после ACK, а при разрыве соединения следующая попытка переподключается
 * и досылает то, что не дошло.
 *
 * Соединением управляет сам транспорт (dtn_exchange): вызывающий не держит
 * дескрипторов и не знает, переподключились мы внутри или нет.
 */

#define DC_DTN_MAGIC      0x4454u   /* 'D','T' */
#define DC_DTN_VERSION    1
#define DC_DTN_HEADER_LEN 18
#define DC_DTN_MAX_BODY   (32 * 1024 * 1024)

#define DC_DTN_FLAG_JSON  0x0001

/* Сколько попыток переподключения внутри одного обмена. */
#define DC_DTN_ATTEMPTS   3

enum {
    DC_DTN_MSG_REQUEST  = 1,
    DC_DTN_MSG_RESPONSE = 2,
    DC_DTN_MSG_PING     = 3,
    DC_DTN_MSG_PONG     = 4,
    DC_DTN_MSG_HELLO    = 5,
    DC_DTN_MSG_WELCOME  = 6,
    DC_DTN_MSG_ACK      = 7,
    DC_DTN_MSG_BUNDLE   = 8
};

typedef struct {
    unsigned char        type;
    unsigned short       flags;
    uint32_t             bundle_id;
    const unsigned char *body;
    size_t               body_len;
} dc_dtn_msg;

/* Очередь неотправленных bundle: каталог, в котором она живёт.
 * Один раз за процесс; повторный вызов с тем же путём ничего не делает. */
int  dtn_queue_init(const char *dir, char *err, size_t errlen);
void dtn_queue_shutdown(void);

/* Сколько bundle ждут отправки прямо сейчас (для отчётов и тестов). */
int  dtn_queue_pending(void);

uint32_t dtn_crc32(const void *data, size_t len);

int dtn_pack(unsigned char type, uint32_t bundle_id, const char *payload, size_t len,
             unsigned char **out, size_t *out_len);

int dtn_unpack(const unsigned char *buf, size_t len, dc_dtn_msg *msg,
               char *err, size_t errlen);

/* Заголовок handshake: клиент шлёт HELLO и ждёт WELCOME. */
int dtn_handshake(int sock, char *err, size_t errlen);

/* Серверная половина того же: читаем HELLO, отвечаем WELCOME. */
int dtn_handshake_accept(int sock, char *err, size_t errlen);

/* Один обмен «отправил — получил ответ», с переподключением при разрыве.
 *   json != NULL — payload это JSON-строка (флаг JSON в заголовке);
 *   json == NULL — payload это len байт сырых данных (команды wt|…, gen|…).
 * Возвращает 0 и кладёт тело ответа в *out_body (его нужно free()).
 * Сокет открывается и закрывается внутри. */
int dtn_exchange(const char *host, int port, unsigned char type,
                 const char *payload, size_t len, long timeout_ms,
                 char **out_body, char *err, size_t errlen);

/* Разбор "dtn://host:port" (также принимает tcp:// и голый host:port). */
int dtn_parse_endpoint(const char *url, char *host, size_t hostlen,
                       int *port, char *err, size_t errlen);

#endif /* DC_DTN_H */