#include "dtn.h"
#include "tcp.h"
#include "util.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* ---------------------------------------------------------------- CRC -- */

uint32_t dtn_crc32(const void *data, size_t len)
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

/* -------------------------------------------------------------- пакет --- */
/*
 * Заголовок 18 байт, big endian:
 *
 *   смещение  размер  поле
 *      0        2     magic = 'D','T'
 *      2        1     версия протокола
 *      3        1     тип сообщения
 *      4        2     флаги (бит 0 — тело помечено как JSON)
 *      6        4     длина тела
 *     10        4     номер bundle
 *     14        4     CRC32 тела
 *     18        N     тело (UTF-8)
 */

int dtn_pack(unsigned char type, uint32_t bundle_id, const char *payload, size_t len,
             unsigned char **out, size_t *out_len)
{
    if (len > DC_DTN_MAX_BODY - DC_DTN_HEADER_LEN) return -1;

    size_t         total = DC_DTN_HEADER_LEN + len;
    unsigned char *buf   = xmalloc(total);

    put_u16(buf, DC_DTN_MAGIC);
    buf[2] = DC_DTN_VERSION;
    buf[3] = type;
    put_u16(buf + 4, 0);
    put_u32(buf + 6, (uint32_t)len);
    put_u32(buf + 10, bundle_id);
    put_u32(buf + 14, len ? dtn_crc32(payload, len) : 0);
    if (len && payload) memcpy(buf + DC_DTN_HEADER_LEN, payload, len);

    *out     = buf;
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
        snprintf(err, errlen, "тело обрезано: ждали %u, получили %zu", body_len,
                 len - DC_DTN_HEADER_LEN);
        return -1;
    }
    if (body_len && dtn_crc32(buf + DC_DTN_HEADER_LEN, body_len) != get_u32(buf + 14)) {
        snprintf(err, errlen, "контрольная сумма не сошлась");
        return -1;
    }
    msg->type      = buf[3];
    msg->flags     = get_u16(buf + 4);
    msg->bundle_id = get_u32(buf + 10);
    msg->body      = buf + DC_DTN_HEADER_LEN;
    msg->body_len  = body_len;
    return 0;
}

/* ------------------------------------------------------------- очередь --- */
/*
 * Имя файла — bundle-<номер>.dtn. Нумерация монотонная, поэтому «что ещё не
 * подтверждено» — это просто файлы, оставшиеся в каталоге.
 */

static char     qdir[512] = "";
static uint32_t next_id   = 1;

static void qpath(char *out, size_t outsz, uint32_t id)
{
    snprintf(out, outsz, "%s/bundle-%u.dtn", qdir, id);
}

int dtn_queue_init(const char *dir, char *err, size_t errlen)
{
    (void)err;
    (void)errlen;
    if (dir && *dir) {
        snprintf(qdir, sizeof qdir, "%s", dir);
        mkdir(qdir, 0755);   /* уже есть — не беда */
    } else {
        qdir[0] = '\0';
    }

    next_id = 1;
    if (!qdir[0]) return 0;

    /* продолжаем нумерацию после остатков прошлого запуска, иначе новые
     * bundle перезапишут неотправленные */
    DIR *d = opendir(qdir);
    if (!d) return 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        unsigned long id = 0;
        if (sscanf(e->d_name, "bundle-%lu.dtn", &id) == 1 && id >= next_id)
            next_id = (uint32_t)id + 1;
    }
    closedir(d);
    return 0;
}

void dtn_queue_shutdown(void)
{
    qdir[0] = '\0';
    next_id = 1;
}

int dtn_queue_pending(void)
{
    if (!qdir[0]) return 0;
    DIR *d = opendir(qdir);
    if (!d) return 0;
    int            n = 0;
    struct dirent *e;
    while ((e = readdir(d)))
        if (strncmp(e->d_name, "bundle-", 7) == 0) n++;
    closedir(d);
    return n;
}

static void qsave(uint32_t id, const unsigned char *data, size_t len)
{
    if (!qdir[0]) return;
    char path[600];
    qpath(path, sizeof path, id);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    if (fwrite(data, 1, len, f) != len) unlink(path);
    fclose(f);
}

static void qdrop(uint32_t id)
{
    if (!qdir[0]) return;
    char path[600];
    qpath(path, sizeof path, id);
    unlink(path);
}

/* -------------------------------------------------------------- чтение --- */
/*
 * Читает ровно одно сообщение. Тело msg->body указывает внутрь *raw, поэтому
 * вызывающий обязан освободить *raw только после того, как перестал читать
 * тело. Так тело не копируется второй раз.
 *
 * Возвращает 1 — прочитано, 0 — poll прерван сигналом (стоит повторить),
 * -1 — ошибка с текстом в err.
 */
static int read_msg(int fd, dc_dtn_msg *msg, unsigned char **raw, long deadline,
                    char *err, size_t errlen)
{
    *raw = NULL;

    long wait = deadline - now_ms();
    if (wait <= 0) {
        snprintf(err, errlen, "таймаут ожидания сообщения");
        return -1;
    }
    struct pollfd pfd = { .fd = fd, .events = POLLIN, .revents = 0 };
    int n = poll(&pfd, 1, (int)wait);
    if (n < 0) {
        if (errno == EINTR) return 0;
        snprintf(err, errlen, "poll: %s", strerror(errno));
        return -1;
    }
    if (n == 0) {
        snprintf(err, errlen, "таймаут ожидания сообщения");
        return -1;
    }

    unsigned char *buf = xmalloc(DC_DTN_HEADER_LEN + 1);
    if (tcp_read_all(fd, buf, DC_DTN_HEADER_LEN, err, errlen) != 0) {
        free(buf);
        return -1;
    }

    /* длину тела читаем до того, как разбираем: недоверенный заголовок не
     * должен заставлять нас выделять память по чужому числу */
    uint32_t body_len = get_u32(buf + 6);
    if (body_len > DC_DTN_MAX_BODY - DC_DTN_HEADER_LEN) {
        free(buf);
        snprintf(err, errlen, "заявленная длина тела %u слишком велика", body_len);
        return -1;
    }
    if (body_len) {
        buf = xrealloc(buf, DC_DTN_HEADER_LEN + body_len + 1);
        wait = deadline - now_ms();
        if (wait <= 0) {
            free(buf);
            snprintf(err, errlen, "таймаут чтения тела");
            return -1;
        }
        pfd.revents = 0;
        n = poll(&pfd, 1, (int)wait);
        if (n < 0) {
            free(buf);
            if (errno == EINTR) return 0;
            snprintf(err, errlen, "poll: %s", strerror(errno));
            return -1;
        }
        if (n == 0) {
            free(buf);
            snprintf(err, errlen, "таймаут чтения тела");
            return -1;
        }
        if (tcp_read_all(fd, buf + DC_DTN_HEADER_LEN, body_len, err, errlen) != 0) {
            free(buf);
            return -1;
        }
    }

    char perr[256] = {0};
    if (dtn_unpack(buf, DC_DTN_HEADER_LEN + body_len, msg, perr, sizeof perr) != 0) {
        free(buf);
        snprintf(err, errlen, "%s", perr);
        return -1;
    }
    *raw = buf;
    return 1;
}

/* ------------------------------------------------------------ handshake --- */

int dtn_handshake(int sock, char *err, size_t errlen)
{
    static const char hello[] = "{\"hello\":\"dtn\"}";
    unsigned char   *msg  = NULL;
    size_t           mlen = 0;
    if (dtn_pack(DC_DTN_MSG_HELLO, 0, hello, sizeof hello - 1, &msg, &mlen) != 0) {
        snprintf(err, errlen, "не удалось упаковать HELLO");
        return -1;
    }
    int rc = tcp_write_all(sock, msg, mlen, err, errlen);
    free(msg);
    if (rc != 0) return -1;

    long deadline = now_ms() + 15000;
    for (;;) {
        dc_dtn_msg   m;
        unsigned char *raw = NULL;
        int r = read_msg(sock, &m, &raw, deadline, err, errlen);
        if (r <= 0) return -1;
        if (m.type == DC_DTN_MSG_WELCOME) {
            free(raw);
            return 0;
        }
        free(raw);
        /* узел мог прислать накопленные bundle раньше WELCOME — пропускаем */
        if (m.type == DC_DTN_MSG_ACK || m.type == DC_DTN_MSG_BUNDLE) continue;
        snprintf(err, errlen, "ожидался WELCOME, пришёл тип %u", m.type);
        return -1;
    }
}

int dtn_handshake_accept(int sock, char *err, size_t errlen)
{
    long deadline = now_ms() + 15000;
    for (;;) {
        dc_dtn_msg    m;
        unsigned char *raw = NULL;
        int r = read_msg(sock, &m, &raw, deadline, err, errlen);
        if (r <= 0) return -1;
        if (m.type == DC_DTN_MSG_HELLO) {
            free(raw);
            static const char welcome[] = "{}";
            unsigned char   *out  = NULL;
            size_t           olen = 0;
            if (dtn_pack(DC_DTN_MSG_WELCOME, 0, welcome, sizeof welcome - 1, &out, &olen) != 0) {
                snprintf(err, errlen, "не удалось упаковать WELCOME");
                return -1;
            }
            int rc = tcp_write_all(sock, out, olen, err, errlen);
            free(out);
            return rc;
        }
        free(raw);
        /* не HELLO — узел не наш, ждать дальше бессмысленно */
        if (m.type == DC_DTN_MSG_WELCOME) {
            snprintf(err, errlen, "получен WELCOME там, где ждали HELLO");
            return -1;
        }
    }
}

/* --------------------------------------------------------------- обмен --- */

static int dial(const char *host, int port, char *err, size_t errlen)
{
    int fd = tcp_socket_client(err, errlen);
    if (fd < 0) return -1;
    if (tcp_connect(fd, host, port, err, errlen) != 0) {
        close(fd);
        return -1;
    }
    if (dtn_handshake(fd, err, errlen) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int dtn_exchange(const char *host, int port, unsigned char type,
                 const char *payload, size_t len, long timeout_ms,
                 char **out_body, char *err, size_t errlen)
{
    if (out_body) *out_body = NULL;

    uint32_t       id  = next_id++;
    unsigned char *msg = NULL;
    size_t         mlen = 0;
    if (dtn_pack(type, id, payload, len, &msg, &mlen) != 0) {
        snprintf(err, errlen, "сообщение не помещается в bundle (%zu байт)", len);
        return -1;
    }
    /* кладём в очередь ДО отправки: обрыв после этого не теряет сообщение */
    qsave(id, msg, mlen);

    long tmo = timeout_ms > 0 ? timeout_ms : 30000;
    long deadline = now_ms() + tmo;
    char derr[512] = {0};

    for (int attempt = 1; attempt <= DC_DTN_ATTEMPTS; attempt++) {
        int fd = dial(host, port, derr, sizeof derr);
        if (fd < 0) {
            if (attempt == DC_DTN_ATTEMPTS) {
                free(msg);
                snprintf(err, errlen, "DTN: не удалось подключиться к %s:%d — %s", host, port,
                         derr);
                return -1;
            }
            continue;
        }

        int written = tcp_write_all(fd, msg, mlen, derr, sizeof derr);
        if (written == 0) {
            for (;;) {
                dc_dtn_msg    m;
                unsigned char *raw = NULL;
                int r = read_msg(fd, &m, &raw, deadline, derr, sizeof derr);
                if (r <= 0) break;

                /* ACK на прошлое сообщение — снимаем его из очереди */
                if (m.type == DC_DTN_MSG_ACK) {
                    qdrop(m.bundle_id);
                    free(raw);
                    continue;
                }
                if (m.type == DC_DTN_MSG_RESPONSE || m.type == DC_DTN_MSG_PONG) {
                    /* подтверждаем приём нашего bundle */
                    unsigned char *ack  = NULL;
                    size_t         alen = 0;
                    if (dtn_pack(DC_DTN_MSG_ACK, id, NULL, 0, &ack, &alen) == 0) {
                        tcp_write_all(fd, ack, alen, NULL, 0);
                        free(ack);
                    }
                    qdrop(id);
                    free(msg);
                    if (out_body && m.body_len)
                        *out_body = xstrndup((const char *)m.body, m.body_len);
                    free(raw);
                    close(fd);
                    return 0;
                }
                free(raw);
            }
        }
        close(fd);
    }

    free(msg);
    snprintf(err, errlen,
             "DTN: нет ответа от %s:%d за %ld мс (%d попытки); bundle %u остался в очереди",
             host, port, tmo, DC_DTN_ATTEMPTS, id);
    return -1;
}

/* --------------------------------------------------------------- адрес --- */

int dtn_parse_endpoint(const char *url, char *host, size_t hostlen, int *port,
                       char *err, size_t errlen)
{
    const char *s = url;
    if (!s || !*s) {
        snprintf(err, errlen, "пустой адрес узла");
        return -1;
    }
    if (strncasecmp(s, "dtn://", 6) == 0)       s += 6;
    else if (strncasecmp(s, "tcp://", 6) == 0)  s += 6;
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
    int   p     = 0;
    if (colon) {
        *colon = '\0';
        p      = atoi(colon + 1);
    }
    if (p <= 0 || p > 65535) {
        snprintf(err, errlen, "в адресе узла нет корректного порта: %s", url);
        return -1;
    }
    *port = p;
    return 0;
}