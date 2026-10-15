#ifndef DC_TCP_H
#define DC_TCP_H

#include <stddef.h>
#include <stdint.h>
#include <netinet/in.h>

/*
 * TCP транспорт планировщик <-> узлы.
 *
 * Использует тот же формат сообщений, что и UDP (14-байтовый заголовок + тело),
 * но поверх надёжного соединения. Преимущества перед UDP:
 *   - нет ограничения на размер сообщения (MTU)
 *   - встроенная проверка целостности (контрольная сумма TCP)
 *   - не нужно переотправлять потерянные пакеты вручную
 *   - меньше накладных расходов при больших задачах
 *
 * Размер сообщения защищем 4-байтовым полем length в заголовке —
 * принимающая сторона читает ровно length байт тела.
 */

#define DC_TCP_MAGIC       0x4443u
#define DC_TCP_VERSION     2
#define DC_TCP_HEADER_LEN  14
#define DC_TCP_MAX_BODY    (64 * 1024 * 1024)  /* 64 МБ — для TCP это нормально */

#define DC_TCP_FLAG_JSON   0x0001

/* Порт по умолчанию для TCP (выделяется рядом с UDP 9101). */
#define DC_TCP_DEFAULT_PORT 9101

enum {
    DC_TCP_MSG_REQUEST  = 1,
    DC_TCP_MSG_RESPONSE = 2,
    DC_TCP_MSG_PING     = 3,
    DC_TCP_MSG_PONG     = 4,
    DC_TCP_MSG_HELLO    = 5,
    DC_TCP_MSG_WELCOME  = 6
};

typedef struct {
    unsigned char       type;
    unsigned short      flags;
    const unsigned char *body;
    size_t              body_len;
} dc_tcp_msg;

uint32_t tcp_crc32(const void *data, size_t len);

/* Упаковывает JSON-тело в TCP-сообщение. Возвращает malloc'нутый буфер. */
int tcp_pack(unsigned char type, const char *json, unsigned char **out, size_t *out_len);

/* Разбирает сообщение. body указывает внутрь buf — копировать не нужно. */
int tcp_unpack(const unsigned char *buf, size_t len, dc_tcp_msg *msg, char *err, size_t errlen);

/* Создаёт TCP-сокет для клиента. */
int tcp_socket_client(char *err, size_t errlen);

/* Создаёт TCP-сокет для сервера (listen). */
int tcp_socket_server(const char *host, int port, char *err, size_t errlen);

/* Подключается к серверу. Возвращает сокет или -1. */
int tcp_connect(int sock, const char *host, int port, char *err, size_t errlen);

/* Принимает одно соединение (accept). */
int tcp_accept(int server_fd, struct sockaddr_in *from, char *err, size_t errlen);

/* Читает ровно len байт из сокета. */
int tcp_read_all(int fd, unsigned char *buf, size_t len, char *err, size_t errlen);

/* Записывает весь буфер в сокет. */
int tcp_write_all(int fd, const unsigned char *buf, size_t len, char *err, size_t errlen);

/* Один запрос-ответ по TCP. Возвращает 0 успехе, тело в *out_body. */
int tcp_request(int sock, const char *host, int port, unsigned char type,
                const char *json, long timeout_ms, char **out_body, char *err, size_t errlen);

/* Как tcp_request, но payload — произвольные данные, а не JSON. */
int tcp_request_raw(int sock, const char *host, int port, const char *payload, size_t len,
                    long timeout_ms, char **out_body, char *err, size_t errlen);

/* Разбирает "tcp://host:port" или "host:port" в адрес для TCP. */
int tcp_parse_endpoint(const char *url, char *host, size_t hostlen, int *port, char *err,
                       size_t errlen);

#endif /* DC_TCP_H */
