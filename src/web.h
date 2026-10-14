#ifndef DC_WEB_H
#define DC_WEB_H

#include "conf.h"

#include <stddef.h>

typedef struct dc_web dc_web;

/* Поднимает HTTP-сервер с панелью управления.
 * api_base — путь API (по умолчанию /api). Демон UDP должен быть уже запущен. */
dc_web *web_start(dc_config *cfg, const char *host, int port, const char *api_base, int no_llm,
                  char *err, size_t errlen);

/* Блокирует вызывающий поток, обслуживая запросы, пока web_stop() не вызван. */
int     web_serve(dc_web *w);
void    web_stop(dc_web *w);
void    web_free(dc_web *w);

int     web_port(const dc_web *w);

/* HTML панели (живёт в ui.c, чтобы web.c не превращался в простыню разметки). */
const char *web_ui_html(void);

#endif /* DC_WEB_H */
