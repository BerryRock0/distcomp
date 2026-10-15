#include "dtn.h"
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
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* ---------------------------------------------------------- состояние --- */

static char   dtn_queue_dir[512] = "";
static uint32_t dtn_next_bundle = 1;
static int    dtn_initialized = 0;

/* Максимум сообщений в очереди на диске. */
#define DC_DTN_MAX_QUEUED 1024

/* Таймаут для одной попытки отправки (мс). */
#define DC_DTN_SEND_TIMEOUT 30000

/* Интервал между попытками флеша очереди (мс). */
#define DC_DTN_FLUSH_INTERVAL 15000

static long dtn_last_flush = 0;

/* ---------------------------------------------------------- утилиты ---- */

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

static uint32_t dtn_crc32(const void *data, size_t len)
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

/* ---------------------------------------------------------- очередь ---- */

static int dtn_queue_path(char *out, size_t outsz, uint32_t bundle_id)
{
    if (!dtn_queue_dir[0]) return -1;
    snprintf(out, outsz, "%s/bundle-%u.dtn", dtn_queue_dir, bundle_id);
    return 0;
}

static int dtn_queue_save(uint32_t bundle_id, const unsigned char *data, size_t len)
{
    char path[600];
    if (dtn_queue_path(path, sizeof path, bundle_id) != 0) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (fwrite(data, 1, len, f) != len) {
        fclose(f);
        unlink(path);
        return -1;
    }
    fclose(f);
    return 0;
}

static int dtn_queue_load(uint32_t bundle_id, unsigned char **out, size_t *out_len)
{
    char path[600];
    if (dtn_queue_path(path, sizeof path, bundle_id) != 0) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(f);
        return -1;
    }
    unsigned char *buf = xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *out = buf;
    *out_len = (size_t)sz;
    return 0;
}

static void dtn_queue_remove(uint32_t bundle_id)
{
    char path[600];
    if (dtn_queue_path(path, sizeof path, bundle_id) != 0) return;
    unlink(path);
}

/* ---------------------------------------------------------- пакетка ---- */

int dtn_pack(unsigned char type, uint32_t bundle_id, const char *json,
             unsigned char **out, size_t *out_len)
{
    size_t body_len = json ? strlen(json) : 0;
    if (body_len > DC_DTN_MAX_BODY - DC_DTN_HEADER_LEN) return -1;

    size_t         total = DC_DTN_HEADER_LEN + body_len;
    unsigned char *buf = xmalloc(total);

    put_u16(buf, DC_DTN_MAGIC);
    buf[2] = DC_DTN_VERSION;
    buf[3] = type;
    put_u16(buf + 4, body_len ? DC_DTN_FLAG_JSON : 0);
    put_u32(buf + 6, (uint32_t)body_len);
    put_u32(buf + 10, bundle_id);
    put_u32(buf + 14, body_len ? dtn_crc32(json, body_len) : 0);
    if (body_len) memcpy(buf + DC_DTN_HEADER_LEN, json, body_len);

    *out = buf;
    *out_len = total;
    return 0;
}

int dtn_unpack(const unsigned char *buf, size_t len, dc_dtn_msg *msg,
               char *err, size_t errlen)
{
    if (len < DC_DTN_HEADER_LEN) {
        snprintf(err, errlen, "сообщение короче заголовка (%zu байт)", len);
        return -1;
    }
    if (get_u16(buf) != DC_DTN_MAGIC) {
        snprintf(err, errlen, "плохая магия — это не DTN-протокол");
        return -1;
    }
    if (buf[2] != DC_DTN_VERSION) {
        snprintf(err, errlen, "версия протокола %u, ожидалась %d", buf[2], DC_DTN_VERSION);
        return -1;
    }
    uint32_t body_len = get_u32(buf + 6);
    if (body_len > DC_DTN_MAX_BODY - DC_DTN_HEADER_LEN) {
        snprintf(err, errlen, "заявленная длина тела %u слишком велика", body_len);
        return -1;
    }
    if (len < DC_DTN_HEADER_LEN + body_len) {
        snprintf(err, errlen, "тело обревано: ждали %u, получили %zu", body_len,
                 len - DC_DTN_HEADER_LEN);
        return -1;
    }
    uint32_t want = get_u32(buf + 14);
    if (body_len && dtn_crc32(buf + DC_DTN_HEADER_LEN, body_len) != want) {
        snprintf(err, errlen, "контрольная сумма не сошлась");
        return -1;
    }
    msg->type = buf[3];
    msg->flags = get_u16(buf + 4);
    msg->bundle_id = get_u32(buf + 10);
    msg->body = buf + DC_DTN_HEADER_LEN;
    msg->body_len = body_len;
    return 0;
}

/* ---------------------------------------------------------- очередь ---- */

int dtn_init(const char *queue_dir, char *err, size_t errlen)
{
    if (dtn_initialized) return 0;
    if (!queue_dir || !*queue_dir) {
        snprintf(err, errlen, "не указан каталог очереди DTN");
        return -1;
    }
    /* Создаём каталог, если его нет */
    if (mkdir(queue_dir, 0755) != 0 && errno != EEXIST) {
        snprintf(err, errlen, "не удалось создать каталог очереди %s: %s", queue_dir,
                 strerror(errno));
        return -1;
    }
    snprintf(dtn_queue_dir, sizeof dtn_queue_dir, "%s", queue_dir);
    dtn_initialized = 1;
    dtn_last_flush = 0;
    log_debug("DTN: очередь в %s", dtn_queue_dir);
    return 0;
}

void dtn_free(void)
{
    dtn_initialized = 0;
    dtn_queue_dir[0] = '\0';
}

void dtn_reset(void)
{
    dtn_next_bundle = 1;
    dtn_last_flush = 0;
}

/* ---------------------------------------------------------- handshake --- */

int dtn_connect(const char *host, int port, char *err, size_t errlen)
{
    int sock = tcp_socket_client(err, errlen);
    if (sock < 0) return -1;
    if (tcp_connect(sock, host, port, err, errlen) != 0) {
        close(sock);
        return -1;
    }

    /* Отправляем HELLO с нашим следующим bundle_id */
    char hello[128];
    snprintf(hello, sizeof hello, "{\"next_bundle\":%u}", dtn_next_bundle);
    unsigned char *msg = NULL;
    size_t         mlen = 0;
    if (dtn_pack(DC_DTN_MSG_HELLO, 0, hello, &msg, &mlen) != 0) {
        snprintf(err, errlen, "не удалось упаковать HELLO");
        close(sock);
        return -1;
    }
    if (tcp_write_all(sock, msg, mlen, err, errlen) != 0) {
        free(msg);
        close(sock);
        return -1;
    }
    free(msg);

    /* Читаем WELCOME */
    unsigned char hdr[DC_DTN_HEADER_LEN];
    struct pollfd pfd = { .fd = sock, .events = POLLIN, .revents = 0 };
    int n = poll(&pfd, 1, 5000);
    if (n <= 0) {
        snprintf(err, errlen, n == 0 ? "таймаут handshake DTN" : "poll: %s", strerror(errno));
        close(sock);
        return -1;
    }
    if (tcp_read_all(sock, hdr, DC_DTN_HEADER_LEN, err, errlen) != 0) {
        close(sock);
        return -1;
    }
    dc_dtn_msg dmsg;
    char       perr[256] = {0};
    if (dtn_unpack(hdr, DC_DTN_HEADER_LEN, &dmsg, perr, sizeof perr) != 0) {
        snprintf(err, errlen, "%s", perr);
        close(sock);
        return -1;
    }
    if (dmsg.type != DC_DTN_MSG_WELCOME) {
        snprintf(err, errlen, "ожидался WELCOME, получили тип %u", dmsg.type);
        close(sock);
        return -1;
    }

    log_debug("DTN: соединение с %s:%d установлено", host, port);
    return sock;
}

int dtn_accept(int server_fd, char *err, size_t errlen)
{
    struct sockaddr_in from;
    int fd = tcp_accept(server_fd, &from, err, errlen);
    if (fd < 0) return -1;

    /* Читаем HELLO */
    unsigned char hdr[DC_DTN_HEADER_LEN];
    struct pollfd pfd = { .fd = fd, .events = POLLIN, .revents = 0 };
    int n = poll(&pfd, 1, 5000);
    if (n <= 0) {
        snprintf(err, errlen, n == 0 ? "таймаут приёма HELLO" : "poll: %s", strerror(errno));
        close(fd);
        return -1;
    }
    if (tcp_read_all(fd, hdr, DC_DTN_HEADER_LEN, err, errlen) != 0) {
        close(fd);
        return -1;
    }
    dc_dtn_msg dmsg;
    char       perr[256] = {0};
    if (dtn_unpack(hdr, DC_DTN_HEADER_LEN, &dmsg, perr, sizeof perr) != 0) {
        snprintf(err, errlen, "%s", perr);
        close(fd);
        return -1;
    }
    if (dmsg.type != DC_DTN_MSG_HELLO) {
        snprintf(err, errlen, "ожидался HELLO, получили тип %u", dmsg.type);
        close(fd);
        return -1;
    }

    /* Отправляем WELCOME */
    unsigned char *msg = NULL;
    size_t         mlen = 0;
    if (dtn_pack(DC_DTN_MSG_WELCOME, 0, "{}", &msg, &mlen) != 0) {
        snprintf(err, errlen, "не удалось упаковать WELCOME");
        close(fd);
        return -1;
    }
    if (tcp_write_all(fd, msg, mlen, err, errlen) != 0) {
        free(msg);
        close(fd);
        return -1;
    }
    free(msg);

    return fd;
}

/* ---------------------------------------------------------- запрос ---- */

int dtn_request(int sock, const char *host, int port, unsigned char type,
                const char *json, long timeout_ms, char **out_body,
                char *err, size_t errlen)
{
    if (out_body) *out_body = NULL;

    uint32_t bundle_id = dtn_next_bundle++;
    unsigned char *msg = NULL;
    size_t         mlen = 0;
    if (dtn_pack(type, bundle_id, json, &msg, &mlen) != 0) {
        snprintf(err, errlen, "не удалось упаковать сообщение");
        return -1;
    }

    /* Сохраняем в очередь перед отправкой */
    if (dtn_queue_dir[0]) {
        dtn_queue_save(bundle_id, msg, mlen);
    }

    long deadline = now_ms() + timeout_ms;
    int  attempt = 0;
    int  max_attempts = 3;

    while (attempt < max_attempts) {
        attempt++;
        long now = now_ms();
        if (now >= deadline) break;

        if (tcp_write_all(sock, msg, mlen, err, errlen) != 0) {
            /* Соединение разорвано — пробуем переподключиться */
            close(sock);
            sock = dtn_connect(host, port, err, errlen);
            if (sock < 0) continue;
        }

        /* Читаем заголовок ответа */
        unsigned char hdr[DC_DTN_HEADER_LEN];
        long wait = deadline - now_ms();
        if (wait <= 0) break;
        struct pollfd pfd = { .fd = sock, .events = POLLIN, .revents = 0 };
        int n = poll(&pfd, 1, (int)(wait > DC_DTN_SEND_TIMEOUT ? DC_DTN_SEND_TIMEOUT : wait));
        if (n < 0) {
            if (errno == EINTR) continue;
            snprintf(err, errlen, "poll: %s", strerror(errno));
            free(msg);
            return -1;
        }
        if (n == 0) continue;  /* таймаут — пробуем снова */

        if (tcp_read_all(sock, hdr, DC_DTN_HEADER_LEN, err, errlen) != 0) {
            close(sock);
            sock = dtn_connect(host, port, err, errlen);
            if (sock < 0) continue;
            continue;
        }

        dc_dtn_msg dmsg;
        char       perr[256] = {0};
        if (dtn_unpack(hdr, DC_DTN_HEADER_LEN, &dmsg, perr, sizeof perr) != 0) {
            snprintf(err, errlen, "%s", perr);
            free(msg);
            return -1;
        }

        /* Пропускаем ACK от предыдущих сообщений */
        if (dmsg.type == DC_DTN_MSG_ACK) {
            dtn_queue_remove(dmsg.bundle_id);
            continue;
        }

        uint32_t body_len = get_u32(hdr + 6);
        unsigned char *body = NULL;
        if (body_len > 0) {
            body = xmalloc(body_len + 1);
            wait = deadline - now_ms();
            if (wait <= 0) {
                free(body);
                break;
            }
            pfd.revents = 0;
            n = poll(&pfd, 1, (int)wait);
            if (n <= 0) {
                free(body);
                continue;
            }
            if (tcp_read_all(sock, body, body_len, err, errlen) != 0) {
                free(body);
                close(sock);
                sock = dtn_connect(host, port, err, errlen);
                if (sock < 0) continue;
                continue;
            }
            body[body_len] = '\0';
        }

        if (dmsg.type != DC_DTN_MSG_RESPONSE && dmsg.type != DC_DTN_MSG_PONG) {
            free(body);
            continue;
        }

        /* Отправляем ACK */
        unsigned char *ack = NULL;
        size_t         alen = 0;
        if (dtn_pack(DC_DTN_MSG_ACK, bundle_id, NULL, &ack, &alen) == 0) {
            tcp_write_all(sock, ack, alen, NULL, 0);
            free(ack);
        }

        dtn_queue_remove(bundle_id);
        free(msg);
        if (out_body && body) *out_body = (char *)body;
        else free(body);
        return 0;
    }

    free(msg);
    snprintf(err, errlen, "DTN: не удалось получить ответ за %ld мс (%d попыток)", timeout_ms, attempt);
    return -1;
}

/* ---------------------------------------------------------- флеш ------ */

void dtn_flush_queue(void)
{
    if (!dtn_initialized || !dtn_queue_dir[0]) return;
    long now = now_ms();
    if (dtn_last_flush > 0 && now - dtn_last_flush < DC_DTN_FLUSH_INTERVAL) return;
    dtn_last_flush = now;

    /* Просто перебираем файлы в каталоге и пытаемся отправить */
    /* В реальной системе здесь была бы проверка адресатов и т.п. */
    /* Для простоты — просто логируем количество */
    /* Полная реализация требует хранения адресата в каждом bundle */
}

/* ---------------------------------------------------------- парсер ----- */

int dtn_parse_endpoint(const char *url, char *host, size_t hostlen,
                       int *port, char *err, size_t errlen)
{
    const char *s = url;
    if (strncasecmp(s, "dtn://", 6) == 0) s += 6;
    else if (strncasecmp(s, "tcp://", 6) == 0) s += 6;
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

    char *colon = strrchr(host, ':');
    int   p = 0;
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
