#include "tcp.h"
#include "util.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

/* ---------------------------------------------------------------- CRC -- */

uint32_t tcp_crc32(const void *data, size_t len)
{
    static uint32_t table[256];
    static int      ready = 0;
    if (!ready) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        ready = 1;
    }
    const unsigned char *p = data;
    uint32_t             c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static void put_u16(unsigned char *p, uint16_t v)
{
    p[0] = (unsigned char)(v >> 8);
    p[1] = (unsigned char)v;
}

static void put_u32(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v >> 24);
    p[1] = (unsigned char)(v >> 16);
    p[2] = (unsigned char)(v >> 8);
    p[3] = (unsigned char)v;
}

static uint16_t get_u16(const unsigned char *p)
{
    return (uint16_t)((p[0] << 8) | p[1]);
}

static uint32_t get_u32(const unsigned char *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

/* ---------------------------------------------------------- упаковка --- */

int tcp_pack(unsigned char type, const char *json, unsigned char **out, size_t *out_len)
{
    size_t body_len = json ? strlen(json) : 0;
    if (body_len > DC_TCP_MAX_BODY - DC_TCP_HEADER_LEN) return -1;

    size_t         total = DC_TCP_HEADER_LEN + body_len;
    unsigned char *buf = xmalloc(total);

    put_u16(buf, DC_TCP_MAGIC);
    buf[2] = DC_TCP_VERSION;
    buf[3] = type;
    put_u16(buf + 4, body_len ? DC_TCP_FLAG_JSON : 0);
    put_u32(buf + 6, (uint32_t)body_len);
    put_u32(buf + 10, body_len ? tcp_crc32(json, body_len) : 0);
    if (body_len) memcpy(buf + DC_TCP_HEADER_LEN, json, body_len);

    *out = buf;
    *out_len = total;
    return 0;
}

int tcp_unpack(const unsigned char *buf, size_t len, dc_tcp_msg *msg, char *err, size_t errlen)
{
    if (len < DC_TCP_HEADER_LEN) {
        snprintf(err, errlen, "сообщение короче заголовка (%zu байт)", len);
        return -1;
    }
    if (get_u16(buf) != DC_TCP_MAGIC) {
        snprintf(err, errlen, "плохая магия — это не наш протокол");
        return -1;
    }
    if (buf[2] != DC_TCP_VERSION) {
        snprintf(err, errlen, "версия протокола %u, ожидалась %d", buf[2], DC_TCP_VERSION);
        return -1;
    }
    uint32_t body_len = get_u32(buf + 6);
    if (body_len > DC_TCP_MAX_BODY - DC_TCP_HEADER_LEN) {
        snprintf(err, errlen, "заявленная длина тела %u слишком велика", body_len);
        return -1;
    }
    if (len < DC_TCP_HEADER_LEN + body_len) {
        snprintf(err, errlen, "тело обревано: ждали %u, получили %zu", body_len,
                 len - DC_TCP_HEADER_LEN);
        return -1;
    }
    uint32_t want = get_u32(buf + 10);
    if (body_len && tcp_crc32(buf + DC_TCP_HEADER_LEN, body_len) != want) {
        snprintf(err, errlen, "контрольная сумма не сошлась");
        return -1;
    }
    msg->type = buf[3];
    msg->flags = get_u16(buf + 4);
    msg->body = buf + DC_TCP_HEADER_LEN;
    msg->body_len = body_len;
    return 0;
}

/* ------------------------------------------------------------ сокеты --- */

int tcp_socket_client(char *err, size_t errlen)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        snprintf(err, errlen, "socket(tcp): %s", strerror(errno));
        return -1;
    }
    /* Уменьшаем задержку для маленьких сообщений */
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    return fd;
}

int tcp_socket_server(const char *host, int port, char *err, size_t errlen)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        snprintf(err, errlen, "socket(tcp): %s", strerror(errno));
        return -1;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (!host || strcmp(host, "0.0.0.0") == 0) {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        snprintf(err, errlen, "неверный адрес привязки: %s", host);
        close(fd);
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) != 0) {
        snprintf(err, errlen, "не удалось занять TCP %s:%d — %s", host ? host : "0.0.0.0", port,
                 strerror(errno));
        close(fd);
        return -1;
    }
    if (listen(fd, 16) != 0) {
        snprintf(err, errlen, "listen %s:%d: %s", host ? host : "0.0.0.0", port, strerror(errno));
        close(fd);
        return -1;
    }
    return fd;
}

int tcp_connect(int sock, const char *host, int port, char *err, size_t errlen)
{
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    char portstr[16];
    snprintf(portstr, sizeof portstr, "%d", port);
    int rc = getaddrinfo(host, portstr, &hints, &res);
    if (rc != 0 || !res) {
        snprintf(err, errlen, "не удалось разрешить %s:%d (%s)", host, port, gai_strerror(rc));
        if (res) freeaddrinfo(res);
        return -1;
    }
    if (res->ai_family != AF_INET) {
        snprintf(err, errlen, "узел %s — не IPv4", host);
        freeaddrinfo(res);
        return -1;
    }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    memcpy(&addr.sin_addr, &((struct sockaddr_in *)res->ai_addr)->sin_addr, sizeof addr.sin_addr);
    freeaddrinfo(res);

    if (connect(sock, (struct sockaddr *)&addr, sizeof addr) != 0) {
        snprintf(err, errlen, "не удалось подключиться к %s:%d: %s", host, port, strerror(errno));
        return -1;
    }
    return 0;
}

int tcp_accept(int server_fd, struct sockaddr_in *from, char *err, size_t errlen)
{
    socklen_t fromlen = from ? sizeof *from : 0;
    int       fd = accept(server_fd, (struct sockaddr *)from, &fromlen);
    if (fd < 0) {
        snprintf(err, errlen, "accept: %s", strerror(errno));
        return -1;
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    return fd;
}

int tcp_read_all(int fd, unsigned char *buf, size_t len, char *err, size_t errlen)
{
    size_t got = 0;
    while (got < len) {
        ssize_t r = recv(fd, buf + got, len - got, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            snprintf(err, errlen, "recv: %s", strerror(errno));
            return -1;
        }
        if (r == 0) {
            snprintf(err, errlen, "соединение закрыто удалённой стороной");
            return -1;
        }
        got += (size_t)r;
    }
    return 0;
}

int tcp_write_all(int fd, const unsigned char *buf, size_t len, char *err, size_t errlen)
{
    size_t sent = 0;
    while (sent < len) {
        ssize_t w = send(fd, buf + sent, len - sent, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            snprintf(err, errlen, "send: %s", strerror(errno));
            return -1;
        }
        if (w == 0) {
            snprintf(err, errlen, "send вернул 0");
            return -1;
        }
        sent += (size_t)w;
    }
    return 0;
}

/* ---------------------------------------------------- запрос-ответ ---- */

int tcp_request(int sock, const char *host, int port, unsigned char type,
                const char *json, long timeout_ms, char **out_body, char *err, size_t errlen)
{
    if (out_body) *out_body = NULL;

    unsigned char *msg = NULL;
    size_t         mlen = 0;
    if (tcp_pack(type, json, &msg, &mlen) != 0) {
        size_t need = json ? strlen(json) : 0;
        snprintf(err, errlen,
                 "задание не влезает в сообщение (%zu байт, максимум %d)",
                 need, DC_TCP_MAX_BODY - DC_TCP_HEADER_LEN);
        return -1;
    }

    long deadline = now_ms() + timeout_ms;
    if (tcp_write_all(sock, msg, mlen, err, errlen) != 0) {
        free(msg);
        return -1;
    }
    free(msg);

    /* Читаем заголовок */
    unsigned char hdr[DC_TCP_HEADER_LEN];
    long wait = deadline - now_ms();
    if (wait <= 0) {
        snprintf(err, errlen, "таймаут %ld мс", timeout_ms);
        return -1;
    }
    struct pollfd pfd = { .fd = sock, .events = POLLIN, .revents = 0 };
    int n = poll(&pfd, 1, (int)wait);
    if (n < 0) {
        if (errno == EINTR) { /* пробуем ещё раз */ }
        else {
            snprintf(err, errlen, "poll: %s", strerror(errno));
            return -1;
        }
    }
    if (n == 0) {
        snprintf(err, errlen, "таймаут %ld мс", timeout_ms);
        return -1;
    }
    if (tcp_read_all(sock, hdr, DC_TCP_HEADER_LEN, err, errlen) != 0) return -1;

    dc_tcp_msg rmsg;
    char       perr[256] = {0};
    if (tcp_unpack(hdr, DC_TCP_HEADER_LEN, &rmsg, perr, sizeof perr) != 0) {
        snprintf(err, errlen, "%s", perr);
        return -1;
    }

    uint32_t body_len = get_u32(hdr + 6);
    unsigned char *body = NULL;
    if (body_len > 0) {
        body = xmalloc(body_len + 1);
        wait = deadline - now_ms();
        if (wait <= 0) {
            free(body);
            snprintf(err, errlen, "таймаут %ld мс", timeout_ms);
            return -1;
        }
        pfd.revents = 0;
        n = poll(&pfd, 1, (int)wait);
        if (n <= 0) {
            free(body);
            if (n == 0) snprintf(err, errlen, "таймаут %ld мс", timeout_ms);
            else snprintf(err, errlen, "poll: %s", strerror(errno));
            return -1;
        }
        if (tcp_read_all(sock, body, body_len, err, errlen) != 0) {
            free(body);
            return -1;
        }
        body[body_len] = '\0';
    }

    if (type == DC_TCP_MSG_PING && rmsg.type != DC_TCP_MSG_PONG) {
        free(body);
        snprintf(err, errlen, "ожидался PONG, получили тип %u", rmsg.type);
        return -1;
    }
    if (rmsg.type != DC_TCP_MSG_RESPONSE && rmsg.type != DC_TCP_MSG_PONG) {
        free(body);
        snprintf(err, errlen, "ожидался RESPONSE, получили тип %u", rmsg.type);
        return -1;
    }

    if (out_body && body) *out_body = (char *)body;
    else free(body);
    return 0;
}

int tcp_request_raw(int sock, const char *host, int port, const char *payload, size_t len,
                    long timeout_ms, char **out_body, char *err, size_t errlen)
{
    if (out_body) *out_body = NULL;

    size_t         total = DC_TCP_HEADER_LEN + len;
    unsigned char *msg = xmalloc(total);

    put_u16(msg, DC_TCP_MAGIC);
    msg[2] = DC_TCP_VERSION;
    msg[3] = DC_TCP_MSG_REQUEST;
    put_u16(msg + 4, 0);
    put_u32(msg + 6, (uint32_t)len);
    put_u32(msg + 10, tcp_crc32(payload, len));
    memcpy(msg + DC_TCP_HEADER_LEN, payload, len);

    long deadline = now_ms() + timeout_ms;
    if (tcp_write_all(sock, msg, total, err, errlen) != 0) {
        free(msg);
        return -1;
    }
    free(msg);

    unsigned char hdr[DC_TCP_HEADER_LEN];
    long wait = deadline - now_ms();
    if (wait <= 0) {
        snprintf(err, errlen, "таймаут %ld мс", timeout_ms);
        return -1;
    }
    struct pollfd pfd = { .fd = sock, .events = POLLIN, .revents = 0 };
    int n = poll(&pfd, 1, (int)wait);
    if (n < 0) {
        if (errno == EINTR) { /* пробуем ещё раз */ }
        else {
            snprintf(err, errlen, "poll: %s", strerror(errno));
            return -1;
        }
    }
    if (n == 0) {
        snprintf(err, errlen, "таймаут %ld мс", timeout_ms);
        return -1;
    }
    if (tcp_read_all(sock, hdr, DC_TCP_HEADER_LEN, err, errlen) != 0) return -1;

    dc_tcp_msg rmsg;
    char       perr[256] = {0};
    if (tcp_unpack(hdr, DC_TCP_HEADER_LEN, &rmsg, perr, sizeof perr) != 0) {
        snprintf(err, errlen, "%s", perr);
        return -1;
    }

    uint32_t body_len = get_u32(hdr + 6);
    unsigned char *body = NULL;
    if (body_len > 0) {
        body = xmalloc(body_len + 1);
        wait = deadline - now_ms();
        if (wait <= 0) {
            free(body);
            snprintf(err, errlen, "таймаут %ld мс", timeout_ms);
            return -1;
        }
        pfd.revents = 0;
        n = poll(&pfd, 1, (int)wait);
        if (n <= 0) {
            free(body);
            if (n == 0) snprintf(err, errlen, "таймаут %ld мс", timeout_ms);
            else snprintf(err, errlen, "poll: %s", strerror(errno));
            return -1;
        }
        if (tcp_read_all(sock, body, body_len, err, errlen) != 0) {
            free(body);
            return -1;
        }
        body[body_len] = '\0';
    }

    if (rmsg.type != DC_TCP_MSG_RESPONSE && rmsg.type != DC_TCP_MSG_PONG) {
        free(body);
        snprintf(err, errlen, "ожидался RESPONSE, получили тип %u", rmsg.type);
        return -1;
    }

    if (out_body && body) *out_body = (char *)body;
    else free(body);
    return 0;
}

int tcp_parse_endpoint(const char *url, char *host, size_t hostlen, int *port, char *err,
                       size_t errlen)
{
    const char *s = url;
    if (strncasecmp(s, "tcp://", 6) == 0) s += 6;
    else if (strncasecmp(s, "http://", 7) == 0) s += 7;
    else if (strncasecmp(s, "https://", 8) == 0) s += 8;

    const char *slash = s + strcspn(s, "/?#");
    size_t      hplen = (size_t)(slash - s);
    if (hplen == 0 || hplen >= hostlen) {
        snprintf(err, errlen, "не удалось разобрать адрес узла: %s", url);
        return -1;
    }
    memcpy(host, s, hplen);
    host[hplen] = '\0';

    char       *colon = strrchr(host, ':');
    int         p = 0;
    if (colon) {
        *colon = '\0';
        p = atoi(colon + 1);
    }
    if (p <= 0 || p > 65535) {
        snprintf(err, errlen, "в адресе узла нет корректного порта: %s", url);
        return -1;
    }
    *port = p;
    return 0;
}
