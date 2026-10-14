#ifndef DC_HTTP_H
#define DC_HTTP_H

#include <stddef.h>

typedef struct {
    int    status;
    char  *body;
    size_t len;
    double seconds;
} http_resp;

void http_resp_free(http_resp *r);

typedef struct {
    long        timeout_ms;
    const char *bearer;        /* -> "Authorization: Bearer <v>" */
    const char *auth_header;   /* custom header name, e.g. "x-api-key" */
    const char *auth_value;    /* its value */
    int         insecure;      /* skip TLS certificate verification */
    const char *extra_header;  /* e.g. "anthropic-version: 2023-06-01" */
} http_opts;

/* Returns 0 on transport success (check r->status for HTTP status), -1 on
 * transport failure (err is filled). */
int http_request(const char *method, const char *url, const char *content_type,
                 const char *body, size_t body_len, const http_opts *o,
                 http_resp *out, char *err, size_t errlen);

#endif /* DC_HTTP_H */
