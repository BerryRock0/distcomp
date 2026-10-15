#ifndef DC_XPORT_H
#define DC_XPORT_H

#include "conf.h"

#include <stddef.h>

/*
 * Выбор транспорта по адресу узла.
 *
 * Адрес узла — это либо «host:port» (тогда UDP, как раньше), либо с явной
 * схемой: udp://, tcp:// или dtn://. Схема решает всё: и какой сокет
 * открывать, и в каком формате уходит сообщение.
 *
 * Раньше каждый вызывающий (планировщик, конвейер модели) повторял один и тот же
 * переключатель «UDP или TCP или DTN». Здесь он один, и добавление транспорта
 * не значит правку во всех местах сразу.
 */

typedef enum {
    DC_TRANS_UDP = 0,   /* по умолчанию: host:port без схемы */
    DC_TRANS_TCP,       /* tcp:// — надёжный поток */
    DC_TRANS_DTN        /* dtn:// — store-and-forward через разрывы */
} dc_transport;

dc_transport xport_from_url(const char *url);

/* Хост и порт из адреса независимо от схемы. */
int xport_parse(const char *url, char *host, size_t hostlen, int *port,
                char *err, size_t errlen);

/* Короткое имя транспорта для отчётов: «UDP», «TCP», «DTN». */
const char *xport_name(dc_transport t);

/* Запрос-ответ; сокет открывается на вызов и закрывается.
 * type — номер сообщения в протоколе DC (DC_MSG_REQUEST, DC_MSG_PING …);
 * для каждого транспорта он переводится в его собственную нумерацию.
 * Тело ответа отдаётся в *out_body (его нужно free()). */
int xport_request(dc_endpoint *ep, unsigned char type, const char *json,
                  long timeout_ms, char **out_body, char *err, size_t errlen);

/* Как xport_request, но payload не JSON: команды узлу shard
 * (wt|…, shape|…, gen|…, part|…) не являются JSON-документом. */
int xport_request_raw(dc_endpoint *ep, const char *payload, size_t len,
                      long timeout_ms, char **out_body, char *err, size_t errlen);

#endif /* DC_XPORT_H */