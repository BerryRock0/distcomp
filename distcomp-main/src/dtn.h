#ifndef DC_DTN_H
#define DC_DTN_H

#include <stddef.h>
#include <stdint.h>

/*
 * DTN (Delay/Disruption-Tolerant Networking) транспорт.
 *
 * В отличие от TCP, DTN предполагает, что соединение может быть:
 *   - с большой задержкой (секунды, минуты)
 *   - с периодическими разрывами
 *   - с высокой потерей пакетов
 *
 * Модель: store-and-forward. Каждое сообщение получает уникальный bundle ID,
 * сохраняется в локальную очередь перед отправкой и удаляется только после
 * получения ACK. Если соединение разрывается — сообщения не теряются,
 * они будут отправлены при следующем подключении.
 *
 * Протокол: поверх TCP, но с подтверждениями и логом неотправленных.
 * При подключении сначала обмениваются накопленными сообщениями.
 */

#define DC_DTN_MAGIC        0x4454u  /* 'DT' */
#define DC_DTN_VERSION      1
#define DC_DTN_HEADER_LEN   18
#define DC_DTN_MAX_BODY     (64 * 1024 * 1024)

#define DC_DTN_FLAG_JSON   0x0001
#define DC_DTN_FLAG_ACK    0x0002

/* Порт по умолчанию для DTN. */
#define DC_DTN_DEFAULT_PORT 9101

enum {
    DC_DTN_MSG_REQUEST   = 1,
    DC_DTN_MSG_RESPONSE  = 2,
    DC_DTN_MSG_PING      = 3,
    DC_DTN_MSG_PONG      = 4,
    DC_DTN_MSG_HELLO     = 5,
    DC_DTN_MSG_WELCOME   = 6,
    DC_DTN_MSG_ACK       = 7,
    DC_DTN_MSG_RETRY     = 8,
    DC_DTN_MSG_BUNDLE    = 9    /* store-and-forward сообщение */
};

typedef struct {
    unsigned char       type;
    unsigned short      flags;
    uint32_t            bundle_id;   /* уникальный ID сообщения для ACK/ретраев */
    const unsigned char *body;
    size_t              body_len;
} dc_dtn_msg;

/* Инициализация DTN-транспорта. queue_dir — каталог для хранения сообщений. */
int  dtn_init(const char *queue_dir, char *err, size_t errlen);
void dtn_free(void);

/* Сброс/инициализация внутреннего состояния без уничтожения очереди. */
void dtn_reset(void);

/* Упаковывает сообщение с bundle_id. */
int  dtn_pack(unsigned char type, uint32_t bundle_id, const char *json,
              unsigned char **out, size_t *out_len);

/* Разбирает сообщение. */
int  dtn_unpack(const unsigned char *buf, size_t len, dc_dtn_msg *msg,
                char *err, size_t errlen);

/* Создаёт TCP-сокет и подключается, затем выполняет DTN handshake.
 * Возвращает сокет или -1. */
int  dtn_connect(const char *host, int port, char *err, size_t errlen);

/* Принимает DTN-соединение (server-side handshake). */
int  dtn_accept(int server_fd, char *err, size_t errlen);

/* Отправляет сообщение и ждёт ACK (с ретраями при необходимости).
 * Возвращает тело ответа в *out_body. */
int  dtn_request(int sock, const char *host, int port, unsigned char type,
                 const char *json, long timeout_ms, char **out_body,
                 char *err, size_t errlen);

/* Проверяет очередь неотправленных сообщений и пытается их доставить.
 * Вызывается периодически из основного цикла. */
void dtn_flush_queue(void);

/* Разбирает "dtn://host:port" в адрес. */
int  dtn_parse_endpoint(const char *url, char *host, size_t hostlen,
                        int *port, char *err, size_t errlen);

#endif /* DC_DTN_H */
