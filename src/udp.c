#include "udp.h"
#include "util.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

/* ---------------------------------------------------------------- CRC -- */

uint32_t udp_crc32(const void *data, size_t len)
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

/* Компактная сборка capability-JSON без strbuf: на маленьком устройстве
 * лишняя аллокация стоит дороже самого сообщения. Значения здесь берутся
 * только из доверенных источников (имя узла, платформа), кавычки не
 * экранируются. Роли в объявлении нет: узел их не знает. */
int udp_pack_hello(int max_body, const char *arch, const char *node,
                   int port, long free_mem, int threads, const char *shard,
                   unsigned char **out, size_t *out_len)
{
    char json[768];
    /* port обязателен: объявление уходит с нового эфемерного сокета, и без
     * него получатель не узнал бы, куда стучаться. threads идёт сюда,
     * чтобы планировщик видел загруженность машины ещё до первого PING. */
    snprintf(json, sizeof json,
             "{\"role\":\"*\",\"node\":\"%s\",\"arch\":\"%s\",\"port\":%d,"
             "\"max_body\":%d,\"free_mem\":%ld,\"threads\":%d,\"shard\":\"%s\","
             "\"proto\":%d}",
             node ? node : "", arch ? arch : "", port, max_body, free_mem, threads,
             shard ? shard : "", DC_UDP_VERSION);
    return udp_pack(DC_MSG_HELLO, json, out, out_len);
}

int udp_pack(unsigned char type, const char *json, unsigned char **out, size_t *out_len)
{
    size_t body_len = json ? strlen(json) : 0;
    if (body_len > DC_UDP_MAX_DGRAM - DC_UDP_HEADER_LEN) return -1;

    size_t         total = DC_UDP_HEADER_LEN + body_len;
    unsigned char *buf = xmalloc(total);

    put_u16(buf, DC_UDP_MAGIC);
    buf[2] = DC_UDP_VERSION;
    buf[3] = type;
    put_u16(buf + 4, body_len ? DC_UDP_FLAG_JSON : 0);
    put_u32(buf + 6, (uint32_t)body_len);
    put_u32(buf + 10, body_len ? udp_crc32(json, body_len) : 0);
    if (body_len) memcpy(buf + DC_UDP_HEADER_LEN, json, body_len);

    *out = buf;
    *out_len = total;
    return 0;
}

int udp_unpack(const unsigned char *buf, size_t len, dc_msg *msg, char *err, size_t errlen)
{
    if (len < DC_UDP_HEADER_LEN) {
        snprintf(err, errlen, "дейтаграмма короче заголовка (%zu байт)", len);
        return -1;
    }
    if (get_u16(buf) != DC_UDP_MAGIC) {
        snprintf(err, errlen, "плохая магия — это не наш протокол");
        return -1;
    }
    if (buf[2] != DC_UDP_VERSION) {
        snprintf(err, errlen, "версия протокола %u, ожидалась %d", buf[2], DC_UDP_VERSION);
        return -1;
    }
    uint32_t body_len = get_u32(buf + 6);
    if (body_len > DC_UDP_MAX_DGRAM - DC_UDP_HEADER_LEN) {
        snprintf(err, errlen, "заявленная длина тела %u слишком велика", body_len);
        return -1;
    }
    if (len < DC_UDP_HEADER_LEN + body_len) {
        snprintf(err, errlen, "тело обревано: ждали %u, получили %zu", body_len,
                 len - DC_UDP_HEADER_LEN);
        return -1;
    }
    uint32_t want = get_u32(buf + 10);
    if (body_len && udp_crc32(buf + DC_UDP_HEADER_LEN, body_len) != want) {
        snprintf(err, errlen, "контрольная сумма не сошлась");
        return -1;
    }
    msg->type = buf[3];
    msg->flags = get_u16(buf + 4);
    msg->body = buf + DC_UDP_HEADER_LEN;
    msg->body_len = body_len;
    return 0;
}

/* ------------------------------------------------------------ сокеты --- */

int udp_socket_client(char *err, size_t errlen)
{
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        snprintf(err, errlen, "socket(udp): %s", strerror(errno));
        return -1;
    }
    /* увеличенный буфер приёма: узел может ответить на пачку запросов,
     * пока планировщик занят другими */
    int rcvbuf = 1 << 20;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof rcvbuf);
    return fd;
}

int udp_socket_bound_ex(const char *host, int port, int rcvbuf, int sndbuf, char *err,
                        size_t errlen)
{
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        snprintf(err, errlen, "socket(udp): %s", strerror(errno));
        return -1;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    setsockopt(fd, SOL_SOCKET, SO_BROADCAST, &one, sizeof one);

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
        snprintf(err, errlen, "не удалось занять UDP %s:%d — %s", host ? host : "0.0.0.0", port,
                 strerror(errno));
        close(fd);
        return -1;
    }
    if (rcvbuf > 0) setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof rcvbuf);
    if (sndbuf > 0) setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof sndbuf);
    return fd;
}

int udp_socket_bound(const char *host, int port, char *err, size_t errlen)
{
    /* увеличенные буферы: узел может отвечать на пачку запросов,
     * пока планировщик занят другими */
    return udp_socket_bound_ex(host, port, 1 << 20, 1 << 20, err, errlen);
}

static int resolve(const char *host, int port, struct sockaddr_in *out, char *err, size_t errlen)
{
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;

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
    memset(out, 0, sizeof *out);
    out->sin_family = AF_INET;
    out->sin_port = htons((unsigned short)port);
    memcpy(&out->sin_addr, &((struct sockaddr_in *)res->ai_addr)->sin_addr, sizeof out->sin_addr);
    freeaddrinfo(res);
    return 0;
}

/* Запрос с произвольным payload (без JSON-флага). */
int udp_request_raw(int sock, const char *host, int port, const char *payload, size_t len,
                    long timeout_ms, char **out_body, char *err, size_t errlen)
{
    if (out_body) *out_body = NULL;
    struct sockaddr_in dst;
    if (resolve(host, port, &dst, err, errlen) != 0) return -1;
    if (connect(sock, (struct sockaddr *)&dst, sizeof dst) != 0) {
        snprintf(err, errlen, "не удалось подключиться к %s:%d: %s", host, port, strerror(errno));
        return -1;
    }
    size_t         total = DC_UDP_HEADER_LEN + len;
    unsigned char *dgram = xmalloc(total);
    put_u16(dgram, DC_UDP_MAGIC);
    dgram[2] = DC_UDP_VERSION;
    dgram[3] = DC_MSG_REQUEST;
    put_u16(dgram + 4, 0);
    put_u32(dgram + 6, (uint32_t)len);
    put_u32(dgram + 10, udp_crc32(payload, len));
    memcpy(dgram + DC_UDP_HEADER_LEN, payload, len);

    long deadline = now_ms() + timeout_ms;
    long retry_every = timeout_ms / 3 > 0 ? timeout_ms / 3 : 50;
    long next_send = 0;
    int  sends = 0;

    for (;;) {
        long now = now_ms();
        if (now >= deadline) {
            free(dgram);
            snprintf(err, errlen, "таймаут %ld мс: узел %s:%d не ответил", timeout_ms, host, port);
            return -1;
        }
        if (now >= next_send) {
            if (send(sock, dgram, total, 0) < 0 && errno != ECONNREFUSED) {
                snprintf(err, errlen, "send %s:%d: %s", host, port, strerror(errno));
                free(dgram);
                return -1;
            }
            sends++;
            next_send = now + retry_every;
            if (sends > 4) next_send = deadline;
        }
        long wait = deadline - now_ms();
        if (wait > retry_every) wait = retry_every;
        struct pollfd pfd = { .fd = sock, .events = POLLIN, .revents = 0 };
        int n = poll(&pfd, 1, (int)wait);
        if (n < 0) {
            if (errno == EINTR) continue;
            snprintf(err, errlen, "poll: %s", strerror(errno));
            free(dgram);
            return -1;
        }
        if (n == 0) continue;
        unsigned char buf[DC_UDP_MAX_DGRAM + 16];
        ssize_t r = recv(sock, buf, sizeof buf, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            if (errno == ECONNREFUSED) {
                snprintf(err, errlen, "узел %s:%d не слушает порт (ICMP unreachable)", host, port);
                free(dgram);
                return -1;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
            snprintf(err, errlen, "recv %s:%d: %s", host, port, strerror(errno));
            free(dgram);
            return -1;
        }
        dc_msg msg;
        char   perr[256] = {0};
        if (udp_unpack(buf, (size_t)r, &msg, perr, sizeof perr) != 0) {
            log_debug("узел %s:%d, битая дейтаграмма отброшена: %s", host, port, perr);
            continue;
        }
        if (msg.type != DC_MSG_RESPONSE && msg.type != DC_MSG_PONG) {
            log_debug("неожиданный тип ответа %u от %s:%d", msg.type, host, port);
            continue;
        }
        free(dgram);
        if (out_body && msg.body_len) *out_body = xstrndup((const char *)msg.body, msg.body_len);
        return 0;
    }
}

int udp_request(int sock, const char *host, int port, unsigned char type,
                const char *json, long timeout_ms, char **out_body, char *err, size_t errlen)
{
    if (out_body) *out_body = NULL;

    struct sockaddr_in dst;
    if (resolve(host, port, &dst, err, errlen) != 0) return -1;

    /*
     * connect() на UDP обязателен: он привязывает сокет к одному адресу, и тогда
     * ядро начинает доставлять ошибки ICMP (в частности port unreachable от
     * закрытого порта). Без этого «мёртвый узел» обнаруживался бы только по
     * истечении полного таймаута, и переключение на резервный узел занимало бы
     * десятки секунд вместо мгновения.
     *
     * Бонус: ядро само отбрасывает дейтаграммы от посторонних отправителей,
     * так что проверку адреса делать не нужно.
     */
    if (connect(sock, (struct sockaddr *)&dst, sizeof dst) != 0) {
        snprintf(err, errlen, "не удалось подключиться к %s:%d: %s", host, port,
                 strerror(errno));
        return -1;
    }

    unsigned char *dgram = NULL;
    size_t         dlen = 0;
    if (udp_pack(type, json, &dgram, &dlen) != 0) {
        size_t need = json ? strlen(json) : 0;
        snprintf(err, errlen,
                 "задание не влезает в одну дейтаграмму (%zu байт, максимум %d) — "
                 "сократите payload или разбейте шаг",
                 need, DC_UDP_MAX_DGRAM - DC_UDP_HEADER_LEN);
        return -1;
    }

    /* повторная отправка уменьшает вероятность потери на загруженной сети */
    long deadline = now_ms() + timeout_ms;
    long retry_every = timeout_ms / 3 > 0 ? timeout_ms / 3 : 50;
    long next_send = 0;
    int  sends = 0;

    for (;;) {
        long now = now_ms();
        if (now >= deadline) {
            free(dgram);
            snprintf(err, errlen, "таймаут %ld мс: узел %s:%d не ответил", timeout_ms, host, port);
            return -1;
        }
        if (now >= next_send) {
            if (send(sock, dgram, dlen, 0) < 0 && errno != ECONNREFUSED) {
                snprintf(err, errlen, "send %s:%d: %s", host, port, strerror(errno));
                free(dgram);
                return -1;
            }
            sends++;
            next_send = now + retry_every;
            if (sends > 4) next_send = deadline; /* дальше пересылать бессмысленно */
        }

        long wait = deadline - now_ms();
        if (wait > retry_every) wait = retry_every;

        struct pollfd pfd = { .fd = sock, .events = POLLIN, .revents = 0 };
        int n = poll(&pfd, 1, (int)wait);
        if (n < 0) {
            if (errno == EINTR) continue;
            snprintf(err, errlen, "poll: %s", strerror(errno));
            free(dgram);
            return -1;
        }
        if (n == 0) continue;

        unsigned char buf[DC_UDP_MAX_DGRAM + 16];
        ssize_t r = recv(sock, buf, sizeof buf, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            if (errno == ECONNREFUSED) {
                snprintf(err, errlen, "узел %s:%d не слушает порт (ICMP unreachable)", host, port);
                free(dgram);
                return -1;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
            snprintf(err, errlen, "recv %s:%d: %s", host, port, strerror(errno));
            free(dgram);
            return -1;
        }

        dc_msg msg;
        char   perr[256] = {0};
        if (udp_unpack(buf, (size_t)r, &msg, perr, sizeof perr) != 0) {
            log_debug("узел %s:%d, битая дейтаграмма отброшена: %s", host, port, perr);
            continue;
        }
        if (type == DC_MSG_PING && msg.type == DC_MSG_PONG) {
            free(dgram);
            if (out_body && msg.body_len) *out_body = xstrndup((const char *)msg.body, msg.body_len);
            return 0;
        }
        if (msg.type != DC_MSG_RESPONSE && msg.type != DC_MSG_PONG) {
            log_debug("неожиданный тип ответа %u от %s:%d", msg.type, host, port);
            continue;
        }

        free(dgram);
        if (out_body) *out_body = xstrndup((const char *)msg.body, msg.body_len);
        return 0;
    }
}

int udp_parse_endpoint(const char *url, char *host, size_t hostlen, int *port, char *err,
                       size_t errlen)
{
    const char *s = url;
    if (strncasecmp(s, "udp://", 6) == 0) s += 6;
    else if (strncasecmp(s, "http://", 7) == 0) s += 7; /* конфиг мог остаться от HTTP-версии */
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
