#ifndef DC_JSON_H
#define DC_JSON_H

#include <stddef.h>

typedef enum {
    JS_NULL,
    JS_BOOL,
    JS_NUM,
    JS_STR,
    JS_ARR,
    JS_OBJ
} js_type;

typedef struct js_value js_value;

struct js_value {
    js_type    type;
    int        bval;
    double     num;
    char      *str;    /* JS_STR: NUL-terminated, decoded */
    char     **keys;   /* JS_OBJ */
    js_value **items;  /* JS_ARR / JS_OBJ */
    size_t     len;
    size_t     cap;
};

typedef struct js_doc js_doc;

js_doc         *js_parse(const char *text, char *err, size_t errlen);
const js_value *js_root(js_doc *doc);
void            js_doc_free(js_doc *doc);

const js_value *js_get(const js_value *obj, const char *key);
const char     *js_get_str(const js_value *obj, const char *key);
const js_value *js_at(const js_value *arr, size_t i);
const char     *js_str(const js_value *v);
double          js_num(const js_value *v, double dflt);
const char     *js_type_name(js_type t);

#endif /* DC_JSON_H */
