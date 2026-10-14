#include "json.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JS_MAX_DEPTH 200

struct js_doc {
    js_value *root;
};

typedef struct {
    const char *p;
    int         depth;
    char       *err;
    size_t      errlen;
    int         failed;
} jctx;

static void jfail(jctx *c, const char *msg)
{
    if (!c->failed && c->err && c->errlen)
        snprintf(c->err, c->errlen, "%s", msg);
    c->failed = 1;
}

static js_value *jnew(js_type t)
{
    js_value *v = xcalloc(1, sizeof *v);
    v->type = t;
    return v;
}

static void jfree(js_value *v)
{
    if (!v) return;
    if (v->type == JS_STR) {
        free(v->str);
    } else if (v->type == JS_ARR) {
        for (size_t i = 0; i < v->len; i++) jfree(v->items[i]);
        free(v->items);
    } else if (v->type == JS_OBJ) {
        for (size_t i = 0; i < v->len; i++) {
            free(v->keys[i]);
            jfree(v->items[i]);
        }
        free(v->keys);
        free(v->items);
    }
    free(v);
}

static void jpush(js_value *v, char *key, js_value *child)
{
    if (v->len == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 8;
        v->items = xrealloc(v->items, v->cap * sizeof *v->items);
        if (v->type == JS_OBJ) v->keys = xrealloc(v->keys, v->cap * sizeof *v->keys);
    }
    if (v->type == JS_OBJ) v->keys[v->len] = key;
    v->items[v->len] = child;
    v->len++;
}

static void jskip_ws(jctx *c)
{
    while (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r') c->p++;
}

static js_value *jparse_value(jctx *c);

static void utf8_put(strbuf *b, unsigned cp)
{
    if (cp < 0x80) {
        sb_addc(b, (char)cp);
    } else if (cp < 0x800) {
        sb_addc(b, (char)(0xC0 | (cp >> 6)));
        sb_addc(b, (char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        sb_addc(b, (char)(0xE0 | (cp >> 12)));
        sb_addc(b, (char)(0x80 | ((cp >> 6) & 0x3F)));
        sb_addc(b, (char)(0x80 | (cp & 0x3F)));
    } else {
        sb_addc(b, (char)(0xF0 | (cp >> 18)));
        sb_addc(b, (char)(0x80 | ((cp >> 12) & 0x3F)));
        sb_addc(b, (char)(0x80 | ((cp >> 6) & 0x3F)));
        sb_addc(b, (char)(0x80 | (cp & 0x3F)));
    }
}

static int jhex4(const char *s, unsigned *out)
{
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        v <<= 4;
        if (c >= '0' && c <= '9')      v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return 0;
    }
    *out = v;
    return 1;
}

/* parses a quoted string starting at c->p == '"', returns decoded text */
static char *jparse_raw_string(jctx *c)
{
    if (*c->p != '"') {
        jfail(c, "expected string");
        return NULL;
    }
    c->p++;
    strbuf b;
    sb_init(&b);
    while (*c->p && *c->p != '"') {
        unsigned char ch = (unsigned char)*c->p;
        if (ch == '\\') {
            c->p++;
            char e = *c->p;
            switch (e) {
            case '"':  sb_addc(&b, '"');  c->p++; break;
            case '\\': sb_addc(&b, '\\'); c->p++; break;
            case '/':  sb_addc(&b, '/');  c->p++; break;
            case 'b':  sb_addc(&b, '\b'); c->p++; break;
            case 'f':  sb_addc(&b, '\f'); c->p++; break;
            case 'n':  sb_addc(&b, '\n'); c->p++; break;
            case 'r':  sb_addc(&b, '\r'); c->p++; break;
            case 't':  sb_addc(&b, '\t'); c->p++; break;
            case 'u': {
                unsigned cp;
                if (!jhex4(c->p + 1, &cp)) {
                    jfail(c, "bad \\u escape");
                    sb_free(&b);
                    return NULL;
                }
                c->p += 5;
                if (cp >= 0xD800 && cp <= 0xDBFF && c->p[0] == '\\' && c->p[1] == 'u') {
                    unsigned lo;
                    if (jhex4(c->p + 2, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        c->p += 6;
                    }
                }
                utf8_put(&b, cp);
                break;
            }
            default:
                jfail(c, "bad escape sequence");
                sb_free(&b);
                return NULL;
            }
        } else if (ch < 0x20) {
            jfail(c, "raw control char in string");
            sb_free(&b);
            return NULL;
        } else {
            sb_addc(&b, (char)ch);
            c->p++;
        }
    }
    if (*c->p != '"') {
        jfail(c, "unterminated string");
        sb_free(&b);
        return NULL;
    }
    c->p++;
    if (!b.p) b.p = xstrdup("");
    return b.p;
}

static js_value *jparse_number(jctx *c)
{
    char *end = NULL;
    double d = strtod(c->p, &end);
    if (end == c->p) {
        jfail(c, "bad number");
        return NULL;
    }
    c->p = end;
    js_value *v = jnew(JS_NUM);
    v->num = d;
    return v;
}

static js_value *jparse_array(jctx *c)
{
    c->p++; /* [ */
    js_value *v = jnew(JS_ARR);
    jskip_ws(c);
    if (*c->p == ']') {
        c->p++;
        return v;
    }
    for (;;) {
        js_value *item = jparse_value(c);
        if (!item) {
            jfree(v);
            return NULL;
        }
        jpush(v, NULL, item);
        jskip_ws(c);
        if (*c->p == ',') {
            c->p++;
            jskip_ws(c);
            continue;
        }
        if (*c->p == ']') {
            c->p++;
            return v;
        }
        jfail(c, "expected ',' or ']'");
        jfree(v);
        return NULL;
    }
}

static js_value *jparse_object(jctx *c)
{
    c->p++; /* { */
    js_value *v = jnew(JS_OBJ);
    jskip_ws(c);
    if (*c->p == '}') {
        c->p++;
        return v;
    }
    for (;;) {
        jskip_ws(c);
        char *key = jparse_raw_string(c);
        if (!key) {
            jfree(v);
            return NULL;
        }
        jskip_ws(c);
        if (*c->p != ':') {
            jfail(c, "expected ':'");
            free(key);
            jfree(v);
            return NULL;
        }
        c->p++;
        js_value *val = jparse_value(c);
        if (!val) {
            free(key);
            jfree(v);
            return NULL;
        }
        jpush(v, key, val);
        jskip_ws(c);
        if (*c->p == ',') {
            c->p++;
            continue;
        }
        if (*c->p == '}') {
            c->p++;
            return v;
        }
        jfail(c, "expected ',' or '}'");
        jfree(v);
        return NULL;
    }
}

static js_value *jparse_value(jctx *c)
{
    if (c->failed) return NULL;
    if (++c->depth > JS_MAX_DEPTH) {
        jfail(c, "nesting too deep");
        c->depth--;
        return NULL;
    }
    jskip_ws(c);
    js_value *v = NULL;
    switch (*c->p) {
    case '{': v = jparse_object(c); break;
    case '[': v = jparse_array(c);  break;
    case '"': {
        char *s = jparse_raw_string(c);
        if (s) {
            v = jnew(JS_STR);
            v->str = s;
        }
        break;
    }
    case 't':
        if (strncmp(c->p, "true", 4) == 0) {
            c->p += 4;
            v = jnew(JS_BOOL);
            v->bval = 1;
        } else jfail(c, "bad literal");
        break;
    case 'f':
        if (strncmp(c->p, "false", 5) == 0) {
            c->p += 5;
            v = jnew(JS_BOOL);
            v->bval = 0;
        } else jfail(c, "bad literal");
        break;
    case 'n':
        if (strncmp(c->p, "null", 4) == 0) {
            c->p += 4;
            v = jnew(JS_NULL);
        } else jfail(c, "bad literal");
        break;
    default:
        if (*c->p == '-' || *c->p == '+' || (*c->p >= '0' && *c->p <= '9'))
            v = jparse_number(c);
        else
            jfail(c, "unexpected character");
    }
    c->depth--;
    return v;
}

js_doc *js_parse(const char *text, char *err, size_t errlen)
{
    if (err && errlen) err[0] = '\0';
    if (!text) {
        if (err && errlen) snprintf(err, errlen, "empty input");
        return NULL;
    }
    jctx c = { .p = text, .depth = 0, .err = err, .errlen = errlen, .failed = 0 };
    js_value *root = jparse_value(&c);
    if (!root) return NULL;
    jskip_ws(&c);
    if (*c.p != '\0') {
        jfail(&c, "trailing garbage after JSON value");
        jfree(root);
        return NULL;
    }
    js_doc *doc = xcalloc(1, sizeof *doc);
    doc->root = root;
    return doc;
}

const js_value *js_root(js_doc *doc) { return doc ? doc->root : NULL; }

void js_doc_free(js_doc *doc)
{
    if (!doc) return;
    jfree(doc->root);
    free(doc);
}

const js_value *js_get(const js_value *obj, const char *key)
{
    if (!obj || obj->type != JS_OBJ || !key) return NULL;
    for (size_t i = 0; i < obj->len; i++)
        if (obj->keys[i] && strcmp(obj->keys[i], key) == 0) return obj->items[i];
    return NULL;
}

const char *js_get_str(const js_value *obj, const char *key)
{
    return js_str(js_get(obj, key));
}

const js_value *js_at(const js_value *arr, size_t i)
{
    if (!arr || (arr->type != JS_ARR && arr->type != JS_OBJ) || i >= arr->len) return NULL;
    return arr->items[i];
}

const char *js_str(const js_value *v)
{
    return (v && v->type == JS_STR) ? v->str : NULL;
}

double js_num(const js_value *v, double dflt)
{
    if (v && v->type == JS_NUM) return v->num;
    if (v && v->type == JS_STR) return strtod(v->str, NULL);
    return dflt;
}

const char *js_type_name(js_type t)
{
    switch (t) {
    case JS_NULL: return "null";
    case JS_BOOL: return "bool";
    case JS_NUM:  return "number";
    case JS_STR:  return "string";
    case JS_ARR:  return "array";
    case JS_OBJ:  return "object";
    }
    return "?";
}
