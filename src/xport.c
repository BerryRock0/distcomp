#include "xport.h"
#include "dtn.h"
#include "tcp.h"
#include "udp.h"
#include "util.h"

#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

dc_transport xport_from_url(const char *url)
{
    if (!url) return DC_TRANS_UDP;
    if (strncasecmp(url, "tcp://", 6) == 0) return DC_TRANS_TCP;
    if (strncasecmp(url, "dtn://", 6) == 0) return DC_TRANS_DTN;
    return DC_TRANS_UDP;
}

const char *xport_name(dc_transport t)
{
    switch (t) {
    case DC_TRANS_TCP: return "TCP";
    case DC_TRANS_DTN: return "DTN";
    default:           return "UDP";
    }
}

int xport_parse(const char *url, char *host, size_t hostlen, int *port,
                char *err, size_t errlen)
{
    if (!url || !*url) {
        snprintf(err, errlen, "пустой адрес узла");
        return -1;
    }
    switch (xport_from_url(url)) {
    case DC_TRANS_TCP: return tcp_parse_endpoint(url, host, hostlen, port, err, errlen);
    case DC_TRANS_DTN: return dtn_parse_endpoint(url, host, hostlen, port, err, errlen);
    default:           return udp_parse_endpoint(url, host, hostlen, port, err, errlen);
    }
}

/* Номера сообщений у трёх транспортов исторически совпадают (REQUEST=1,
 * PING=3 …), но полагаться на это нельзя: переводим явно. */
static unsigned char msg_for(dc_transport t, unsigned char type, int raw)
{
    if (raw) return type;   /* у произвольного payload тип всегда REQUEST */
    switch (t) {
    case DC_TRANS_TCP:
        return (unsigned char)(type == DC_MSG_PING ? DC_TCP_MSG_PING
                             : type == DC_MSG_REQUEST ? DC_TCP_MSG_REQUEST
                                                       : type);
    case DC_TRANS_DTN:
        return (unsigned char)(type == DC_MSG_PING ? DC_DTN_MSG_PING
                             : type == DC_MSG_REQUEST ? DC_DTN_MSG_REQUEST
                                                       : type);
    default:
        return type;
    }
}

/* DTN подтверждение стоит дороже обычного обмена: канал может быть узким,
 * поэтому на PING (health-check) даём ему больше времени, чем на шаг. */
static long timeout_for(dc_transport t, long timeout_ms, unsigned char type)
{
    if (t == DC_TRANS_DTN && type == DC_MSG_PING && timeout_ms < 10000)
        return 10000;
    return timeout_ms;
}

static int request_common(dc_endpoint *ep, unsigned char type, const char *json,
                          const char *raw, size_t raw_len, long timeout_ms,
                          char **out_body, char *err, size_t errlen)
{
    if (!ep || !ep->url) {
        snprintf(err, errlen, "не указан узел");
        return -1;
    }
    if (out_body) *out_body = NULL;

    char host[320];
    int  port = 0;
    if (xport_parse(ep->url, host, sizeof host, &port, err, errlen) != 0) return -1;

    dc_transport t     = xport_from_url(ep->url);
    unsigned char wire = msg_for(t, type, raw != NULL);
    long          tmo  = timeout_for(t, timeout_ms, type);

    switch (t) {
    case DC_TRANS_TCP: {
        int s = tcp_socket_client(err, errlen);
        if (s < 0) return -1;
        if (tcp_connect(s, host, port, err, errlen) != 0) {
            close(s);
            return -1;
        }
        int rc = raw ? tcp_request_raw(s, host, port, raw, raw_len, tmo, out_body, err,
                                       errlen)
                     : tcp_request(s, host, port, wire, json, tmo, out_body, err, errlen);
        close(s);
        return rc;
    }
    case DC_TRANS_DTN: {
        /* DTN держит соединение внутри: открывает, переподключает при обрыве
         * и закрывает сам — наружу отдаётся только тело ответа. */
        const char *body   = raw ? raw : json;
        size_t      blen   = raw ? raw_len : (json ? strlen(json) : 0);
        return dtn_exchange(host, port, wire, body, blen, tmo, out_body, err, errlen);
    }
    default: {
        int s = udp_socket_client(err, errlen);
        if (s < 0) return -1;
        int rc = raw ? udp_request_raw(s, host, port, raw, raw_len, tmo, out_body, err,
                                       errlen)
                     : udp_request(s, host, port, wire, json, tmo, out_body, err, errlen);
        close(s);
        return rc;
    }
    }
}

int xport_request(dc_endpoint *ep, unsigned char type, const char *json,
                  long timeout_ms, char **out_body, char *err, size_t errlen)
{
    return request_common(ep, type, json, NULL, 0, timeout_ms, out_body, err, errlen);
}

int xport_request_raw(dc_endpoint *ep, const char *payload, size_t len,
                      long timeout_ms, char **out_body, char *err, size_t errlen)
{
    return request_common(ep, DC_MSG_REQUEST, NULL, payload, len, timeout_ms, out_body, err,
                          errlen);
}