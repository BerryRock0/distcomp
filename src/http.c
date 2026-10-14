#include "http.h"
#include "conf.h"    /* DC_VERSION в User-Agent */
#include "util.h"

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <openssl/err.h>
#include <openssl/ssl.h>

#define HTTP_MAX_BODY (64UL * 1024 * 1024)

void http_resp_free(http_resp *r)
{
    if (!r) return;
    free(r->body);
    r->body = NULL;
    r->len = 0;
}

static long remaining_ms(long deadline)
{
    long left = deadline - now_ms();
    return left > 0 ? left : 1;
}

static void set_nonblock(int fd)
{
    int fl = fcntl(fd, F_GETFL, 0);
    if (fl >= 0) fcntl(fd, F_SETFL, fl | O_NONBLOCK);
}

static int wait_fd(int fd, int events, long deadline, char *err, size_t errlen)
{
    struct pollfd pfd = { .fd = fd, .events = (short)events, .revents = 0 };
    for (;;) {
        int n = poll(&pfd, 1, (int)remaining_ms(deadline));
        if (n < 0) {
            if (errno == EINTR) continue;
            snprintf(err, errlen, "poll: %s", strerror(errno));
            return -1;
        }
        if (n == 0) {
            snprintf(err, errlen, "таймаут ожидания ответа");
            return -1;
        }
        return 0;
    }
}

/* ------------------------------------------------------------------ URL -- */

typedef struct {
    char scheme[8];
    char host[320];
    char port[8];
    char path[2048];
} url_parts;

static int url_parse(const char *url, url_parts *u, char *err, size_t errlen)
{
    memset(u, 0, sizeof *u);
    const char *sep = strstr(url, "://");
    if (!sep) {
        snprintf(err, errlen, "неверный URL (нет схемы): %s", url);
        return -1;
    }
    size_t sl = (size_t)(sep - url);
    if (sl >= sizeof u->scheme) {
        snprintf(err, errlen, "схема слишком длинная");
        return -1;
    }
    memcpy(u->scheme, url, sl);
    u->scheme[sl] = '\0';
    str_lower(u->scheme);
    if (strcmp(u->scheme, "http") && strcmp(u->scheme, "https")) {
        snprintf(err, errlen, "поддерживаются только http и https");
        return -1;
    }

    const char *hostport = sep + 3;
    const char *slash = hostport + strcspn(hostport, "/?#");
    size_t hplen = (size_t)(slash - hostport);
    if (hplen == 0 || hplen >= sizeof u->host) {
        snprintf(err, errlen, "неверный хост в URL: %s", url);
        return -1;
    }
    char hp[sizeof u->host];
    memcpy(hp, hostport, hplen);
    hp[hplen] = '\0';

    char *colon = strrchr(hp, ':');
    char *rbrack = strchr(hp, ']');
    /* скобочный IPv6 — [::1]:8080: двоеточия внутри скобок это адрес,
     * порт отделяется только двоеточием после закрывающей скобки */
    if (colon && (rbrack == NULL || colon > rbrack)) {
        *colon = '\0';
        snprintf(u->port, sizeof u->port, "%s", colon + 1);
    } else {
        snprintf(u->port, sizeof u->port, "%s", strcmp(u->scheme, "https") == 0 ? "443" : "80");
    }
    snprintf(u->host, sizeof u->host, "%s", hp);
    snprintf(u->path, sizeof u->path, "%s", (*slash == '/') ? slash : "/");
    if (!u->path[0]) snprintf(u->path, sizeof u->path, "/");
    /* query string that came after a bare path is already part of u->path */
    return 0;
}

/* --------------------------------------------------------------- conn --- */

typedef struct {
    int      fd;
    SSL     *ssl;
    SSL_CTX *ctx;
    long     deadline;
} conn;

static void conn_close(conn *c)
{
    if (c->ssl) {
        SSL_shutdown(c->ssl);
        SSL_free(c->ssl);
        c->ssl = NULL;
    }
    if (c->ctx) {
        SSL_CTX_free(c->ctx);
        c->ctx = NULL;
    }
    if (c->fd >= 0) {
        close(c->fd);
        c->fd = -1;
    }
}

static int conn_tcp_connect(const char *host, const char *port, long deadline,
                            char *err, size_t errlen)
{
    struct addrinfo hints, *res = NULL, *ai;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    int rc = getaddrinfo(host, port, &hints, &res);
    if (rc != 0) {
        snprintf(err, errlen, "не удалось разрешить %s:%s (%s)", host, port, gai_strerror(rc));
        return -1;
    }
    int fd = -1;
    char lasterr[256] = "нет адресов";
    for (ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            snprintf(lasterr, sizeof lasterr, "socket: %s", strerror(errno));
            continue;
        }
        set_nonblock(fd);
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);

        int r = connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (r == 0) break;
        if (errno != EINPROGRESS) {
            snprintf(lasterr, sizeof lasterr, "connect: %s", strerror(errno));
            close(fd);
            fd = -1;
            continue;
        }
        if (wait_fd(fd, POLLOUT, deadline, err, errlen) != 0) {
            close(fd);
            fd = -1;
            continue;
        }
        int soerr = 0;
        socklen_t sl = sizeof soerr;
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &sl) != 0 || soerr != 0) {
            snprintf(lasterr, sizeof lasterr, "connect: %s", strerror(soerr ? soerr : errno));
            close(fd);
            fd = -1;
            continue;
        }
        break;
    }
    freeaddrinfo(res);
    if (fd < 0) {
        snprintf(err, errlen, "не удалось подключиться к %s:%s (%s)", host, port, lasterr);
        return -1;
    }
    return fd;
}

static int conn_tls_start(conn *c, const char *host, int insecure,
                          char *err, size_t errlen)
{
    static int inited = 0;
    if (!inited) {
        SSL_library_init();
        SSL_load_error_strings();
        inited = 1;
    }
    c->ctx = SSL_CTX_new(TLS_client_method());
    if (!c->ctx) {
        snprintf(err, errlen, "SSL_CTX_new failed");
        return -1;
    }
    SSL_CTX_set_min_proto_version(c->ctx, TLS1_2_VERSION);
    if (insecure) {
        SSL_CTX_set_verify(c->ctx, SSL_VERIFY_NONE, NULL);
    } else {
        SSL_CTX_set_default_verify_paths(c->ctx);
        SSL_CTX_set_verify(c->ctx, SSL_VERIFY_PEER, NULL);
    }
    c->ssl = SSL_new(c->ctx);
    if (!c->ssl) {
        snprintf(err, errlen, "SSL_new failed");
        return -1;
    }
    SSL_set_tlsext_host_name(c->ssl, host);          /* SNI */
    if (!insecure) SSL_set1_host(c->ssl, host);      /* проверка имени в сертификате */
    SSL_set_fd(c->ssl, c->fd);
    SSL_set_connect_state(c->ssl);

    for (;;) {
        int r = SSL_do_handshake(c->ssl);
        if (r == 1) return 0;
        int e = SSL_get_error(c->ssl, r);
        if (e == SSL_ERROR_WANT_READ) {
            if (wait_fd(c->fd, POLLIN, c->deadline, err, errlen) != 0) return -1;
        } else if (e == SSL_ERROR_WANT_WRITE) {
            if (wait_fd(c->fd, POLLOUT, c->deadline, err, errlen) != 0) return -1;
        } else {
            unsigned long se = ERR_get_error();
            char msg[256] = "неизвестная ошибка TLS";
            if (se) ERR_error_string_n(se, msg, sizeof msg);
            snprintf(err, errlen, "TLS handshake failed: %s", msg);
            return -1;
        }
    }
}

static int conn_write(conn *c, const void *buf, size_t n, char *err, size_t errlen)
{
    const char *p = buf;
    size_t left = n;
    while (left > 0) {
        ssize_t w;
        if (c->ssl) {
            w = SSL_write(c->ssl, p, (int)(left > INT32_MAX ? INT32_MAX : left));
            if (w <= 0) {
                int e = SSL_get_error(c->ssl, (int)w);
                if (e == SSL_ERROR_WANT_WRITE) {
                    if (wait_fd(c->fd, POLLOUT, c->deadline, err, errlen) != 0) return -1;
                    continue;
                }
                if (e == SSL_ERROR_WANT_READ) {
                    if (wait_fd(c->fd, POLLIN, c->deadline, err, errlen) != 0) return -1;
                    continue;
                }
                snprintf(err, errlen, "SSL_write failed");
                return -1;
            }
        } else {
            w = send(c->fd, p, left, MSG_NOSIGNAL);
            if (w < 0) {
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                    if (wait_fd(c->fd, POLLOUT, c->deadline, err, errlen) != 0) return -1;
                    continue;
                }
                snprintf(err, errlen, "send: %s", strerror(errno));
                return -1;
            }
        }
        p += w;
        left -= (size_t)w;
    }
    return 0;
}

/* returns >0 bytes, 0 on clean EOF, -1 on error */
static ssize_t conn_read(conn *c, void *buf, size_t n, char *err, size_t errlen)
{
    for (;;) {
        ssize_t r;
        if (c->ssl) {
            int got = SSL_read(c->ssl, buf, (int)(n > INT32_MAX ? INT32_MAX : n));
            if (got > 0) return got;
            int e = SSL_get_error(c->ssl, got);
            if (e == SSL_ERROR_WANT_READ) {
                if (wait_fd(c->fd, POLLIN, c->deadline, err, errlen) != 0) return -1;
                continue;
            }
            if (e == SSL_ERROR_WANT_WRITE) {
                if (wait_fd(c->fd, POLLOUT, c->deadline, err, errlen) != 0) return -1;
                continue;
            }
            if (e == SSL_ERROR_ZERO_RETURN) return 0;
            if (got == 0) return 0; /* обрыв без close_notify */
            snprintf(err, errlen, "SSL_read failed");
            return -1;
        }
        r = recv(c->fd, buf, n, 0);
        if (r > 0) return r;
        if (r == 0) return 0;
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            if (wait_fd(c->fd, POLLIN, c->deadline, err, errlen) != 0) return -1;
            continue;
        }
        snprintf(err, errlen, "recv: %s", strerror(errno));
        return -1;
    }
}

/* ------------------------------------------------------------ response -- */

static int header_get(const char *head, const char *name, char *out, size_t outsz)
{
    if (outsz) out[0] = '\0';
    size_t nl = strlen(name);
    const char *line = strchr(head, '\n');
    if (!line) return 0;
    line++;
    while (*line && *line != '\r' && *line != '\n') {
        const char *eol = strchr(line, '\n');
        size_t linelen = eol ? (size_t)(eol - line) : strlen(line);
        if (linelen > nl && strncasecmp(line, name, nl) == 0 && line[nl] == ':') {
            const char *v = line + nl + 1;
            while (v < line + linelen && (*v == ' ' || *v == '\t')) v++;
            size_t vl = (size_t)(line + linelen - v);
            if (vl >= outsz) vl = outsz - 1;
            memcpy(out, v, vl);
            out[vl] = '\0';
            size_t t = strlen(out);
            while (t > 0 && (out[t - 1] == '\r' || out[t - 1] == ' ')) out[--t] = '\0';
            return 1;
        }
        if (!eol) break;
        line = eol + 1;
    }
    return 0;
}

static int read_chunked(conn *c, strbuf *out, char *err, size_t errlen)
{
    char line[128];
    for (;;) {
        size_t li = 0;
        for (;;) {
            char ch;
            ssize_t r = conn_read(c, &ch, 1, err, errlen);
            if (r <= 0) {
                snprintf(err, errlen, "обрыв при чтении chunk-размера");
                return -1;
            }
            if (ch == '\n') break;
            if (ch != '\r' && li + 1 < sizeof line) line[li++] = ch;
        }
        line[li] = '\0';
        char *semi = strchr(line, ';');
        if (semi) *semi = '\0';
        long size = strtol(line, NULL, 16);
        if (size <= 0) break;
        /* сначала клампим size, потом вычитаем: out->len + size может
         * переполнить size_t и проверка прошла бы молча */
        if ((size_t)size > HTTP_MAX_BODY || out->len > HTTP_MAX_BODY - (size_t)size) {
            snprintf(err, errlen, "ответ слишком велик");
            return -1;
        }
        char *tmp = xmalloc((size_t)size);
        size_t got = 0;
        while (got < (size_t)size) {
            ssize_t r = conn_read(c, tmp + got, (size_t)size - got, err, errlen);
            if (r <= 0) {
                free(tmp);
                snprintf(err, errlen, "обрыв в теле chunk");
                return -1;
            }
            got += (size_t)r;
        }
        sb_addn(out, tmp, got);
        free(tmp);
        /* CRLF после данных; обрыв здесь — ошибка, а не повод читать
         * следующий chunk-size и ругаться на «обрыв при чтении размера» */
        char crlf[2];
        if (conn_read(c, crlf, 2, err, errlen) != 2) {
            snprintf(err, errlen, "обрыв после тела chunk");
            return -1;
        }
    }
    /* трейлеры */
    for (;;) {
        char line2[256];
        size_t li = 0;
        int got_nl = 0;
        for (;;) {
            char ch;
            ssize_t r = conn_read(c, &ch, 1, err, errlen);
            if (r <= 0) break;
            if (ch == '\n') { got_nl = 1; break; }
            if (li + 1 < sizeof line2) line2[li++] = ch;
        }
        line2[li] = '\0';
        if (li <= 2 || !got_nl) break;
    }
    return 0;
}

int http_request(const char *method, const char *url, const char *content_type,
                 const char *body, size_t body_len, const http_opts *o,
                 http_resp *out, char *err, size_t errlen)
{
    long t0 = now_ms();
    memset(out, 0, sizeof *out);

    url_parts u;
    if (url_parse(url, &u, err, errlen) != 0) return -1;

    long timeout = (o && o->timeout_ms > 0) ? o->timeout_ms : 30000;
    int insecure = o ? o->insecure : 0;

    conn c = { .fd = -1, .ssl = NULL, .ctx = NULL, .deadline = t0 + timeout };

    c.fd = conn_tcp_connect(u.host, u.port, c.deadline, err, errlen);
    if (c.fd < 0) return -1;

    if (strcmp(u.scheme, "https") == 0) {
        if (conn_tls_start(&c, u.host, insecure, err, errlen) != 0) {
            conn_close(&c);
            return -1;
        }
    }

    strbuf req;
    sb_init(&req);
    sb_printf(&req, "%s %s HTTP/1.1\r\n", method, u.path);
    sb_printf(&req, "Host: %s:%s\r\n", u.host, u.port);
    sb_add(&req, "User-Agent: distcomp/" DC_VERSION "\r\n");
    sb_add(&req, "Accept: application/json\r\n");
    sb_add(&req, "Connection: close\r\n");
    if (content_type) sb_printf(&req, "Content-Type: %s\r\n", content_type);
    if (o && o->bearer && *o->bearer)
        sb_printf(&req, "Authorization: Bearer %s\r\n", o->bearer);
    if (o && o->auth_header && o->auth_value)
        sb_printf(&req, "%s: %s\r\n", o->auth_header, o->auth_value);
    if (o && o->extra_header && *o->extra_header)
        sb_printf(&req, "%s\r\n", o->extra_header);
    sb_printf(&req, "Content-Length: %zu\r\n\r\n", body_len);

    if (conn_write(&c, req.p, req.len, err, errlen) != 0) goto fail;
    if (body_len && conn_write(&c, body, body_len, err, errlen) != 0) goto fail;

    strbuf raw;
    sb_init(&raw);
    size_t head_end = 0;
    for (;;) {
        char *nlp = raw.len >= 4 ? memmem(raw.p, raw.len, "\r\n\r\n", 4) : NULL;
        if (nlp) {
            head_end = (size_t)(nlp - raw.p) + 4;
            break;
        }
        if (raw.len > 128 * 1024) {
            snprintf(err, errlen, "слишком большие HTTP-заголовки");
            sb_free(&raw);
            goto fail;
        }
        char buf[8192];
        ssize_t r = conn_read(&c, buf, sizeof buf, err, errlen);
        if (r < 0) {
            sb_free(&raw);
            goto fail;
        }
        if (r == 0) {
            if (raw.len == 0) {
                snprintf(err, errlen, "соединение закрыто без ответа");
                sb_free(&raw);
                goto fail;
            }
            break; /* ответ без завершающих заголовков */
        }
        sb_addn(&raw, buf, (size_t)r);
    }

    char *head = raw.p ? xstrndup(raw.p, head_end) : xstrdup("");
    if (strncmp(head, "HTTP/", 5) != 0) {
        snprintf(err, errlen, "некорректная строка статуса");
        free(head);
        sb_free(&raw);
        goto fail;
    }
    const char *sp = strchr(head, ' ');
    out->status = sp ? atoi(sp + 1) : 0;

    strbuf bodybuf;
    sb_init(&bodybuf);
    char te[128] = {0}, cl[128] = {0};
    header_get(head, "Transfer-Encoding", te, sizeof te);
    header_get(head, "Content-Length", cl, sizeof cl);

    if (te[0] && strcasestr(te, "chunked")) {
        if (read_chunked(&c, &bodybuf, err, errlen) != 0) {
            free(head);
            sb_free(&raw);
            sb_free(&bodybuf);
            goto fail;
        }
    } else if (cl[0]) {
        size_t want = (size_t)strtoul(cl, NULL, 10);
        if (want > HTTP_MAX_BODY) {
            snprintf(err, errlen, "ответ слишком велик (%zu байт)", want);
            free(head);
            sb_free(&raw);
            sb_free(&bodybuf);
            goto fail;
        }
        if (head_end < raw.len) sb_addn(&bodybuf, raw.p + head_end, raw.len - head_end);
        while (bodybuf.len < want) {
            char buf[8192];
            ssize_t r = conn_read(&c, buf, sizeof buf, err, errlen);
            if (r <= 0) break;
            sb_addn(&bodybuf, buf, (size_t)r);
        }
        if (bodybuf.len > want) bodybuf.len = want;
    } else {
        if (head_end < raw.len) sb_addn(&bodybuf, raw.p + head_end, raw.len - head_end);
        for (;;) {
            char buf[8192];
            ssize_t r = conn_read(&c, buf, sizeof buf, err, errlen);
            if (r <= 0) break;
            sb_addn(&bodybuf, buf, (size_t)r);
        }
    }

    if (!bodybuf.p) bodybuf.p = xstrdup("");
    out->body = bodybuf.p;
    out->len = bodybuf.len;
    out->seconds = (double)(now_ms() - t0) / 1000.0;

    free(head);
    sb_free(&raw);
    sb_free(&req);
    conn_close(&c);
    return 0;

fail:
    sb_free(&req);
    conn_close(&c);
    return -1;
}
