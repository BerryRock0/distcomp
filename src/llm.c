#include "llm.h"
#include "expr.h"
#include "http.h"
#include "json.h"
#include "plan.h"
#include "util.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------ промпт --- */

static void describe_role(const dc_config *cfg, const char *name, const char *what,
                          strbuf *b)
{
    const dc_role *r = conf_role((dc_config *)cfg, name, 0);
    sb_printf(b, "- \"%s\" (%s) — %s; обслуживают ", name, what, "вычислительные узлы");
    if (r && r->nep) {
        for (int i = 0; i < r->nep; i++) {
            if (i) sb_add(b, ", ");
            sb_printf(b, "%s", r->eps[i].url);
            /* Лимит тела известен из PING: без него модель нарежет шаг на 8 КБ
             * и не поймёт, почему плата его не выполнила */
            if (r->eps[i].cap.known && r->eps[i].cap.max_body > 0)
                sb_printf(b, " (примет не больше %d байт задачи)", r->eps[i].cap.max_body);
        }
    } else {
        sb_add(b, "нет назначенных узлов");
    }
    sb_addc(b, '\n');
}

/* Промпт зависит от того, что удалось узнать об узлах, а возможности
 * уточняются по ходу работы. Поэтому в ключ кэша входит и число узлов,
 * про которые мы уже что-то знаем: иначе после PONG показался бы старый
 * промпт — без лимитов тела. */
static int caps_fingerprint(const dc_config *cfg)
{
    int n = 0;
    for (int i = 0; i < cfg->nroles; i++)
        for (int j = 0; j < cfg->roles[i].nep; j++)
            if (cfg->roles[i].eps[j].cap.known) n++;
    return n;
}

const char *llm_system_prompt(const dc_config *cfg)
{
    static char *cache = NULL;
    static const dc_config *cache_cfg = NULL;
    static int   cache_expr = -1;
    static int   cache_caps = -1;
    int expr_mode = cfg->payload_mode && str_ieq(cfg->payload_mode, "expr");
    int caps = caps_fingerprint(cfg);
    if (cache && cache_cfg == cfg && cache_expr == expr_mode && cache_caps == caps)
        return cache;
    strbuf b;
    sb_init(&b);

    sb_add(&b,
        "Ты — планировщик распределённых вычислений (распределитель задач).\n"
        "Твоя единственная задача: разложить цель пользователя на минимальный набор шагов\n"
        "и назначить каждому шагу роль. Выполнять шаги будешь не ты, а узлы на удалённых\n"
        "машинах — сам выбирай машину не нужно, это делает планировщик по загрузке.\n"
        "\n"
        "ДОСТУПНЫЕ РОЛИ (использовать только их):\n");
    describe_role(cfg, ROLE_COMPUTE,
                  "арифметика и формулы, выдаёт числовой результат", &b);
    describe_role(cfg, ROLE_VERIFY,
                  "проверка уже полученных результатов на непротиворечивость", &b);
    describe_role(cfg, ROLE_MEMORIZE,
                  "финальная запись итогового результата в постоянную память", &b);

    sb_add(&b,
        "\nФОРМАТ ОТВЕТА — строго один JSON-объект, без markdown, без комментариев, без текста вокруг:\n"
        "{\n"
        "  \"goal\": \"цель одной строкой\",\n"
        "  \"steps\": [\n"
        "    {\"id\":\"s1\",\"role\":\"compute\",\"title\":\"что делает шаг\","
        "\"payload\":\"что именно передать узлу\",\"deps\":[]}\n"
        "  ]\n"
        "}\n"
        "\nЖЁСТКИЕ ПРАВИЛА:\n"
        "1. Ответ — только JSON. Никаких ``` ограждений.\n"
        "2. id уникальны, зависят только от уже существующих id, граф ацикличен.\n"
        "3. Порядок по смыслу: сначала compute, затем verify, затем memorize.\n"
        "4. Шагов от 2 до 8. Не выдумывай шаги, которые не нужны для цели.\n"
        "5. deps — массив строк, может быть пустым.\n");

    if (expr_mode) {
        sb_add(&b,
            "6. ВАЖНО: compute-узел умеет считать арифметические выражения.\n"
            "   Поэтому payload для роли \"compute\" — это ГОТОВОЕ выражение,\n"
            "   например \"(1250*3+480)/7\" или \"sqrt(144)+2^5\".\n"
            "   Никакого текста, единиц измерения, слов «штуки» и т.п. — только выражение.\n"
            "   Если из цели нельзя вывести выражение — всё равно составь корректное\n"
            "   выражение из чисел, которые есть в цели.\n"
            "7. payload для \"verify\" — арифметическое выражение, результат которого\n"
            "   должен совпасть с результатом compute-шагов; либо слово \"all\",\n"
            "   чтобы проверить, что все результаты непустые.\n"
            "8. payload для \"memorize\" — краткое описание того, что сохранить\n"
            "   (например \"итоговая сумма закупки\").\n");
    } else {
        sb_add(&b,
            "6. payload — произвольный текст: это задание для узла соответствующей роли.\n"
            "   Для \"compute\" сформулируй конкретную вычислительную задачу.\n"
            "   Для \"verify\" — конкретную проверку.\n"
            "   Для \"memorize\" — что именно сохранить.\n");
    }

    sb_add(&b,
        "\nПример (для цели «посчитать стоимость закупки: 1250*3 плюс доставка 480, "
        "разделить на 7 частей»):\n"
        "{\"goal\":\"стоимость закупки и её доля\",\"steps\":[\n"
        " {\"id\":\"s1\",\"role\":\"compute\",\"title\":\"стоимость закупки\",\"payload\":\"1250*3\",\"deps\":[]},\n"
        " {\"id\":\"s2\",\"role\":\"compute\",\"title\":\"доля одной части\",\"payload\":\"(1250*3+480)/7\",\"deps\":[]},\n"
        " {\"id\":\"s3\",\"role\":\"verify\",\"title\":\"сверка с суммой\",\"payload\":\"all\",\"deps\":[\"s1\",\"s2\"]},\n"
        " {\"id\":\"s4\",\"role\":\"memorize\",\"title\":\"итог\",\"payload\":\"стоимость закупки и доля части\",\"deps\":[\"s3\"]}]}\n");

    free(cache);
    cache = b.p;
    cache_cfg = cfg;
    cache_expr = expr_mode;
    cache_caps = caps;
    return cache;
}

/* ---------------------------------------------------------- запрос --- */

/* Хост из URL: нужно, чтобы отличить локальный сервер (llama.cpp или Ollama
 * на этой же машине либо в своей сети) от облачного, где запрос без ключа
 * бесполезен — и сказать об этом сразу, а не получить 401 от сервера. */
static int llm_host_is_local(const char *url)
{
    char host[320] = "";
    int  port = 0;
    char err[192] = {0};

    const char *p = strstr(url, "://");
    p = p ? p + 3 : url;
    const char *slash = p + strcspn(p, "/?#");
    size_t      n = (size_t)(slash - p);
    char       *colon = memchr(p, ':', n);
    if (colon) n = (size_t)(colon - p);
    if (n == 0 || n >= sizeof host) return 0;
    memcpy(host, p, n);
    host[n] = '\0';

    (void)port;
    (void)err;
    if (strcmp(host, "localhost") == 0 || strcmp(host, "::1") == 0) return 1;
    unsigned a, b2, c, d2;
    if (sscanf(host, "%u.%u.%u.%u", &a, &b2, &c, &d2) == 4) {
        if (a == 127) return 1;
        if (a == 10) return 1;
        if (a == 192 && b2 == 168) return 1;
        if (a == 172 && b2 >= 16 && b2 <= 31) return 1;
        if (a == 169 && b2 == 254) return 1;
    }
    if (strstr(host, ".local") || strstr(host, ".lan") || strstr(host, ".home")) return 1;
    return 0;
}

int llm_has_key(const dc_config *cfg)
{
    return (cfg->llm_key && *cfg->llm_key) || getenv("OPENAI_API_KEY") ||
           getenv("ANTHROPIC_API_KEY") ? 1 : 0;
}

int llm_needs_key(const dc_config *cfg)
{
    const char *mode = cfg->llm_auth ? cfg->llm_auth : "auto";
    if (str_ieq(mode, "none")) return 0;
    if (str_ieq(mode, "bearer")) return 1;
    return llm_has_key(cfg) ? 0 : llm_host_is_local(cfg->llm_url) ? 0 : 1;
}

/* Отправлять ли заголовок авторизации. Это не то же самое, что «нужен ли
 * ключ»: ключ, заданный в конфигурации, отправляется в любом случае — в том
 * числе локальному серверу, запущенному с --api-key. */
static int llm_send_key(const dc_config *cfg)
{
    const char *mode = cfg->llm_auth ? cfg->llm_auth : "auto";
    if (str_ieq(mode, "none")) return 0;
    return llm_has_key(cfg);
}

static char *build_request(const dc_config *cfg, const char *goal, const char *rejected,
                           const char *reason, int json_mode)
{
    const char *sys = llm_system_prompt(cfg);
    strbuf b;
    sb_init(&b);
    int anthropic = str_ieq(cfg->llm_style, "anthropic");

    sb_add(&b, "{ \"model\": ");
    sb_add_json_string(&b, cfg->llm_model);
    sb_printf(&b, ", \"temperature\": %.2f", cfg->llm_temp);
    if (anthropic) {
        sb_add(&b, ", \"max_tokens\": 4096, \"system\": ");
        sb_add_json_string(&b, sys);
        sb_add(&b, ", \"messages\": [{\"role\":\"user\",\"content\":");
        sb_add_json_string(&b, goal);
        if (rejected) {
            /* модель видит свой прошлый ответ целиком: просить исправить
             * «вот это» куда точнее, чем просить «сделай лучше» */
            sb_add(&b, "},{\"role\":\"assistant\",\"content\":");
            sb_add_json_string(&b, rejected);
            sb_add(&b, "},{\"role\":\"user\",\"content\":");
            sb_add_json_string(&b, reason);
        }
        sb_add(&b, "}]");
    } else {
        sb_add(&b, ", \"messages\": [{\"role\":\"system\",\"content\":");
        sb_add_json_string(&b, sys);
        sb_add(&b, "},{\"role\":\"user\",\"content\":");
        sb_add_json_string(&b, goal);
        if (rejected) {
            sb_add(&b, "},{\"role\":\"assistant\",\"content\":");
            sb_add_json_string(&b, rejected);
            sb_add(&b, "},{\"role\":\"user\",\"content\":");
            sb_add_json_string(&b, reason);
        }
        sb_add(&b, "}]");
        if (json_mode) sb_add(&b, ", \"response_format\": {\"type\":\"json_object\"}");
    }
    sb_add(&b, " }");
    return b.p;
}

/* ---------------------------------------------------- разбор ответа --- */

static char *content_to_text(const js_value *v)
{
    if (!v) return NULL;
    if (v->type == JS_STR) return xstrdup(v->str);
    if (v->type == JS_ARR) {
        strbuf b;
        sb_init(&b);
        for (size_t i = 0; i < v->len; i++) {
            const js_value *it = js_at(v, i);
            const char     *t = js_get_str(it, "text");
            if (t) sb_add(&b, t);
        }
        return b.p ? b.p : xstrdup("");
    }
    return NULL;
}

static char *extract_content(const char *style, const char *body, char *err, size_t errlen)
{
    char     perr[256] = {0};
    js_doc  *doc = js_parse(body, perr, sizeof perr);
    if (!doc) {
        snprintf(err, errlen, "ответ LLM не JSON: %s", perr);
        return NULL;
    }
    const js_value *root = js_root(doc);
    char           *out = NULL;

    /* ответ может прийти в виде {"error":{"message":...}} даже со статусом 200 —
     * часть прокси так делает; тогда важно показать текст ошибки, а не «пусто» */
    if (js_get(root, "error")) {
        const char     *m = js_get_str(js_get(root, "error"), "message");
        snprintf(err, errlen, "%s: %s", style, m ? m : "тело ответа без описания ошибки");
        js_doc_free(doc);
        return NULL;
    }

    if (str_ieq(style, "anthropic")) {
        const js_value *c = js_get(root, "content");
        if (c) out = content_to_text(c);
        if (!out || !*out) snprintf(err, errlen, "%s: пустой content", style);
    } else {
        const js_value *choices = js_get(root, "choices");
        const js_value *choice = js_at(choices, 0);
        const js_value *msg = js_get(choice, "message");
        if (msg) out = content_to_text(js_get(msg, "content"));
        if (!out) {
            /* Responses API: output_text */
            const js_value *ot = js_get(root, "output_text");
            if (ot) out = content_to_text(ot);
        }
        if (!out || !*out) {
            const char *fr = js_get_str(choice, "finish_reason");
            /* length — самая частая и самая обидная причина: модель начала
             * рассуждение и не уложилась в max_tokens, плана в ответе нет */
            if (fr && str_ieq(fr, "length"))
                snprintf(err, errlen,
                         "%s: ответ обрезан по длине (finish_reason=length) — "
                         "упростите цель или поднимите лимит токенов", style);
            else
                snprintf(err, errlen, "%s: пустой choices[].message.content%s%s", style,
                         fr ? ", finish_reason=" : "", fr ? fr : "");
        }
    }
    js_doc_free(doc);
    return out;
}

/* снимает markdown-ограждение и обрезает всё лишнее до { ... } */
static char *sanitize_json_text(const char *text)
{
    strbuf b;
    sb_init(&b);

    const char *start = strstr(text, "```");
    if (start) {
        start += 3;
        if (strncmp(start, "json", 4) == 0) {
            while (*start && *start != '\n') start++;
            start++;
        }
        const char *end = strstr(start, "```");
        sb_addn(&b, start, end ? (size_t)(end - start) : strlen(start));
    } else {
        sb_add(&b, text);
    }

    const char *ob = strchr(b.p, '{');
    const char *cb = b.p ? strrchr(b.p, '}') : NULL;
    if (!ob || !cb || cb < ob) {
        sb_free(&b);
        return NULL;
    }
    char *out = xstrndup(ob, (size_t)(cb - ob + 1));
    sb_free(&b);
    return out;
}

/* ------------------------------------------------- эвристический план --- */

static int is_expr_char(char c)
{
    return isdigit((unsigned char)c) || c == '.' || c == '+' || c == '-' || c == '*' ||
           c == '/' || c == '%' || c == '^' || c == '(' || c == ')' || c == ' ' ||
           c == '\t' || c == ',';
}

/* достаёт из цели максимальные куски, похожие на арифметику, и оставляет
 * только те, которые реально считаются */
static char *extract_expressions(const char *goal)
{
    strbuf b;
    sb_init(&b);
    int seen_any = 0;
    const char *p = goal;

    while (*p) {
        if (!is_expr_char(*p)) {
            p++;
            continue;
        }
        const char *start = p;
        while (*p && is_expr_char(*p)) p++;
        char *cand = xstrndup(start, (size_t)(p - start));

        /* обрезаем пробелы и запятые по краям */
        char *t = str_trim(cand);
        char *copy = xstrdup(t);
        for (char *q = copy; *q; q++)
            if (*q == ',') *q = ' ';
        free(cand);

        double  val = 0;
        char    eerr[128] = {0};
        int     has_digit = 0, has_op = 0;
        for (char *q = copy; *q; q++) {
            if (isdigit((unsigned char)*q)) has_digit = 1;
            if (*q == '+' || *q == '-' || *q == '*' || *q == '/' || *q == '^') has_op = 1;
        }
        if (has_digit && has_op && expr_eval(copy, &val, eerr, sizeof eerr) == 0) {
            if (seen_any) sb_add(&b, "\n");
            sb_add(&b, copy);
            seen_any = 1;
        }
        free(copy);
    }
    if (!b.p) return NULL;
    return b.p;
}

char *llm_heuristic_plan(const dc_config *cfg, const char *goal)
{
    int   expr_mode = !cfg->payload_mode || str_ieq(cfg->payload_mode, "expr");
    char *exprs = expr_mode ? extract_expressions(goal) : NULL;
    int   have_verify   = conf_role((dc_config *)cfg, ROLE_VERIFY, 0) != NULL;
    int   have_memorize = conf_role((dc_config *)cfg, ROLE_MEMORIZE, 0) != NULL;

    char *expr[64];
    int   ne = 0;
    if (exprs) {
        char *line = strtok(exprs, "\n");
        while (line && ne < 64) {
            expr[ne++] = line;
            line = strtok(NULL, "\n");
        }
    }
    if (ne == 0) {
        free(exprs);
        exprs = NULL;
    }

    /* собираем шаги в список строк, затем склеиваем запятыми */
    char *parts[70];
    int   np = 0;

    if (ne > 0) {
        for (int i = 0; i < ne; i++) {
            strbuf s;
            sb_init(&s);
            sb_printf(&s,
                      "{\"id\":\"c%d\",\"role\":\"%s\",\"title\":\"вычислить %s\","
                      "\"payload\":\"%s\",\"deps\":[]}",
                      i + 1, ROLE_COMPUTE, expr[i], expr[i]);
            parts[np++] = s.p;
        }
    } else {
        strbuf s;
        sb_init(&s);
        sb_printf(&s, "{\"id\":\"c1\",\"role\":\"%s\",\"title\":\"выполнить цель\",\"payload\":",
                  ROLE_COMPUTE);
        sb_add_json_string(&s, goal);
        sb_add(&s, ",\"deps\":[]}");
        parts[np++] = s.p;
        ne = 1;
    }

    if (have_verify) {
        strbuf s;
        sb_init(&s);
        sb_printf(&s,
                  "{\"id\":\"v1\",\"role\":\"%s\",\"title\":\"проверить результаты\","
                  "\"payload\":\"all\",\"deps\":[", ROLE_VERIFY);
        for (int i = 0; i < ne; i++) sb_printf(&s, "\"c%d\"%s", i + 1, (i + 1 < ne) ? "," : "");
        sb_add(&s, "]}");
        parts[np++] = s.p;
    }
    if (have_memorize) {
        strbuf s;
        sb_init(&s);
        sb_printf(&s,
                  "{\"id\":\"m1\",\"role\":\"%s\",\"title\":\"сохранить итог\",\"payload\":",
                  ROLE_MEMORIZE);
        sb_add_json_string(&s, goal);
        sb_add(&s, ",\"deps\":[");
        if (have_verify) {
            sb_add(&s, "\"v1\"");
        } else {
            for (int i = 0; i < ne; i++)
                sb_printf(&s, "\"c%d\"%s", i + 1, (i + 1 < ne) ? "," : "");
        }
        sb_add(&s, "]}");
        parts[np++] = s.p;
    }

    strbuf b;
    sb_init(&b);
    sb_add(&b, "{\"goal\": ");
    sb_add_json_string(&b, goal);
    sb_add(&b, ", \"steps\": [\n");
    for (int i = 0; i < np; i++) sb_printf(&b, "    %s%s\n", parts[i], i + 1 < np ? "," : "");
    sb_add(&b, "  ]}\n");

    for (int i = 0; i < np; i++) free(parts[i]);
    free(exprs);
    return b.p;
}

/* Один поход к модели: вернуть JSON плана либо NULL с причиной.
 * rejected/reason — прошлый ответ и причина, по которой он не подошёл;
 * на первой попытке они NULL. */
static char *llm_ask(const dc_config *cfg, const char *goal, const char *rejected,
                     const char *reason, char *err, size_t errlen)
{
    int json_mode = !cfg->llm_json_mode || str_ieq(cfg->llm_json_mode, "auto")
                        ? !str_ieq(cfg->llm_style, "anthropic")
                        : str_ieq(cfg->llm_json_mode, "on");

    char *req = build_request(cfg, goal, rejected, reason, json_mode);

    /* локальному серверу заголовок авторизации не шлём вообще: некоторые
     * из них (llama.cpp) отвечают 401 на лишний Bearer, даже когда ключа нет */
    int need_key = llm_send_key(cfg);
    http_opts o = {
        .timeout_ms = cfg->llm_timeout_ms,
        .bearer = need_key ? cfg->llm_key : NULL,
        .auth_header = need_key ? cfg->llm_auth_header : NULL,
        .auth_value = need_key ? cfg->llm_key : NULL,
        .insecure = getenv("DC_INSECURE") ? atoi(getenv("DC_INSECURE")) : 0,
        .extra_header = cfg->llm_extra_header,
    };

    http_resp resp;
    char      herr[512] = {0};
    log_info("запрос к LLM: %s (модель %s)%s", cfg->llm_url, cfg->llm_model,
             reason ? ", с уточнением" : "");
    int rc = http_request("POST", cfg->llm_url, "application/json", req, strlen(req), &o, &resp,
                          herr, sizeof herr);
    free(req);

    if (rc != 0) {
        snprintf(err, errlen, "не удалось обратиться к LLM: %s", herr);
        return NULL;
    }
    if (resp.status < 200 || resp.status >= 300) {
        snprintf(err, errlen, "LLM ответила HTTP %d: %.300s", resp.status,
                 resp.body ? resp.body : "");
        http_resp_free(&resp);
        return NULL;
    }

    char eerr[512] = {0};
    char *text = extract_content(cfg->llm_style, resp.body ? resp.body : "", eerr, sizeof eerr);
    http_resp_free(&resp);
    if (!text) {
        snprintf(err, errlen, "%s", eerr);
        return NULL;
    }

    char *clean = sanitize_json_text(text);
    free(text);
    if (!clean) {
        snprintf(err, errlen, "в ответе LLM не найден JSON-объект с планом");
        return NULL;
    }
    log_debug("план от LLM: %s", clean);
    return clean;
}

static int llm_available(const dc_config *cfg)
{
    if (!cfg->llm_url || !*cfg->llm_url || !cfg->llm_model || !*cfg->llm_model) return 0;
    /* локальному серверу ключ не нужен: llama.cpp server, Ollama и LM Studio
     * запускаются вовсе без него, и требование ключа отсекало их на входе */
    return !llm_needs_key(cfg);
}

/* Просьба исправить план: короткая, адресная, без угроз. Модели нужны и
 * её собственный прошлый ответ, и точная причина отказа. */
static char *repair_prompt(const char *reason)
{
    strbuf b;
    sb_init(&b);
    sb_add(&b, "Твой план отклонён проверкой: ");
    sb_add(&b, reason);
    sb_add(&b,
           "\nВерни исправленный план целиком, тем же JSON-форматом и без "
           "пояснений вокруг. Учти замечания: роли должны быть из списка "
           "допустимых, id — уникальными, зависимости — только на уже "
           "объявленные id, граф — без циклов.");
    return b.p;
}

char *llm_make_plan(const dc_config *cfg, const char *goal, int use_llm, int allow_fallback,
                    char *err, size_t errlen)
{
    if (!use_llm) {
        log_info("режим --no-llm: строю план эвристикой");
        return llm_heuristic_plan(cfg, goal);
    }

    if (!llm_available(cfg)) {
        if (!allow_fallback) {
            if (cfg->llm_url && *cfg->llm_url && !llm_has_key(cfg) &&
                !str_ieq(cfg->llm_auth ? cfg->llm_auth : "auto", "none"))
                snprintf(err, errlen,
                         "сервер %s не локальный, а ключ не задан. Либо укажите "
                         "api_key = $OPENAI_API_KEY, либо поставьте auth = none, "
                         "если это ваш сервер в своей сети",
                         cfg->llm_url);
            else
                snprintf(err, errlen,
                         "LLM не настроена: задайте [llm] url и model; ключ нужен "
                         "только для облака (api_key = $OPENAI_API_KEY). Локальному "
                         "серверу — llama.cpp, Ollama, LM Studio — ключ не нужен");
            return NULL;
        }
        log_info("LLM не настроена — использую эвристический планировщик");
        return llm_heuristic_plan(cfg, goal);
    }

    /*
     * Модель ошибается не только на сети: чаще всего план приходит
     * негодным по существу — цикл в зависимостях, роль не из списка,
     * обрезанный ответ. Молча откатываться на эвристику тут нельзя:
     * пользователь просил LLM, и получил бы не то, что просил, не заметив.
     * Поэтому показываем модели её собственный ответ и причину отказа и
     * даём chance исправить. Эвристика остаётся последним рубежом.
     */
    char *rejected = NULL;
    char  reason[1024] = "";
    char  lasterr[1280] = "";   /* с запасом на префикс «план от LLM отклонён: » */

    for (int attempt = 1; attempt <= DC_LLM_TRIES; attempt++) {
        char aerr[1024] = {0};
        char *json = llm_ask(cfg, goal, rejected, rejected ? reason : NULL, aerr, sizeof aerr);
        if (!json) {
            log_error("%s", aerr);
            snprintf(lasterr, sizeof lasterr, "%s", aerr);
            break;   /* до модели не дотянулись — повтор не поможет */
        }

        char  perr[1024] = {0};
        plan *p = plan_from_json(json, cfg, perr, sizeof perr);
        if (p) {
            plan_free(p);
            free(rejected);
            return json;
        }

        log_error("план от LLM отклонён: %s", perr);
        free(rejected);
        rejected = json;                    /* теперь это «прошлый ответ» */
        char *rp = repair_prompt(perr);
        snprintf(reason, sizeof reason, "%s", rp ? rp : "");
        free(rp);
        snprintf(lasterr, sizeof lasterr, "план от LLM отклонён: %s", perr);

        if (attempt < DC_LLM_TRIES) log_info("прошу модель исправить план (попытка %d)", attempt + 1);
    }

    free(rejected);
    snprintf(err, errlen, "%s", lasterr[0] ? lasterr : "LLM не вернула пригодный план");
    if (!allow_fallback) return NULL;
    log_info("перехожу на эвристический планировщик");
    return llm_heuristic_plan(cfg, goal);
}

/* -------------------------------------------------- проверка LLM --- */

/* Результат проверки сервера. Один раз собирается, дальше его читают и CLI,
 * и веб-панель — иначе они показывали бы разное. */
typedef struct {
    int  rc;           /* 0 — сервер ответил и модель на месте */
    int  needs_key;
    int  has_key;
    char models_url[512];
    char error[512];
    char *models[32];
    int  nmodels;
} llm_status;

static void llm_status_free(llm_status *st)
{
    for (int i = 0; i < st->nmodels; i++) free(st->models[i]);
    st->nmodels = 0;
}

/* Список моделей сервера: {"data":[{"id":...}]} (OpenAI) или
 * {"models":[{"name":...}]} (Ollama в старом виде). */
static void llm_status_models(llm_status *st, const char *body)
{
    char      perr[256] = {0};
    js_doc   *d = js_parse(body && *body ? body : "", perr, sizeof perr);
    if (!d) return;

    const js_value *list = js_get(js_root(d), "data");
    if (!list || list->type != JS_ARR) list = js_get(js_root(d), "models");
    if (!list || list->type != JS_ARR) {
        js_doc_free(d);
        return;
    }
    for (size_t i = 0; i < list->len && st->nmodels < 32; i++) {
        const js_value *it = js_at(list, i);
        const char     *id = js_get_str(it, "id");
        if (!id) id = js_get_str(it, "name");
        if (id && *id) st->models[st->nmodels++] = xstrdup(id);
    }
    js_doc_free(d);
}

static int llm_probe(const dc_config *cfg, llm_status *st)
{
    memset(st, 0, sizeof *st);
    st->has_key = llm_has_key(cfg);
    st->needs_key = llm_needs_key(cfg);

    if (!cfg->llm_url || !*cfg->llm_url) {
        snprintf(st->error, sizeof st->error, "не задан [llm] url");
        return st->rc = 3;
    }

    const char *slash = strrchr(cfg->llm_url, '/');
    size_t      base = slash ? (size_t)(slash - cfg->llm_url) : strlen(cfg->llm_url);
    /* у .../v1/chat/completions снимаем два последних сегмента: /v1/models */
    if (base >= 5 && strncmp(cfg->llm_url + base - 5, "/chat", 5) == 0) base -= 5;
    snprintf(st->models_url, sizeof st->models_url, "%.*s/models", (int)base, cfg->llm_url);

    if (st->needs_key && !st->has_key) {
        snprintf(st->error, sizeof st->error,
                 "сервер %s не локальный, а ключ не задан: задайте api_key "
                 "или auth = none", cfg->llm_url);
        return st->rc = 3;
    }

    http_opts o = {
        .timeout_ms = cfg->llm_timeout_ms > 0 ? cfg->llm_timeout_ms : 10000,
        .bearer = llm_send_key(cfg) ? cfg->llm_key : NULL,
        .auth_header = llm_send_key(cfg) ? cfg->llm_auth_header : NULL,
        .auth_value = llm_send_key(cfg) ? cfg->llm_key : NULL,
        .insecure = getenv("DC_INSECURE") ? atoi(getenv("DC_INSECURE")) : 0,
        .extra_header = cfg->llm_extra_header,
    };
    http_resp resp;
    char      herr[512] = {0};
    int       rc = http_request("GET", st->models_url, NULL, NULL, 0, &o, &resp, herr, sizeof herr);
    if (rc != 0) {
        snprintf(st->error, sizeof st->error, "сервер не отвечает: %s", herr);
        return st->rc = 3;
    }
    if (resp.status == 401 || resp.status == 403) {
        snprintf(st->error, sizeof st->error,
                 "сервер требует авторизацию (HTTP %d): задайте api_key или "
                 "auth = none, если сервер локальный", resp.status);
        http_resp_free(&resp);
        return st->rc = 3;
    }
    if (resp.status < 200 || resp.status >= 300) {
        snprintf(st->error, sizeof st->error, "сервер ответил HTTP %d: %.200s", resp.status,
                 resp.body ? resp.body : "");
        http_resp_free(&resp);
        return st->rc = 3;
    }

    llm_status_models(st, resp.body);
    http_resp_free(&resp);

    if (st->nmodels == 0) {
        snprintf(st->error, sizeof st->error,
                 "сервер ответил, но список моделей разобрать не удалось — скорее всего "
                 "это не OpenAI-совместимый endpoint: %s", cfg->llm_url);
        return st->rc = 3;
    }
    if (cfg->llm_model && *cfg->llm_model) {
        int ours = 0;
        for (int i = 0; i < st->nmodels; i++)
            if (strcmp(st->models[i], cfg->llm_model) == 0) ours = 1;
        if (!ours) {
            snprintf(st->error, sizeof st->error,
                     "модели \"%s\" на сервере нет — укажите в [llm] model ту, "
                     "что в списке", cfg->llm_model);
            return st->rc = 3;
        }
    }
    return st->rc = 0;
}

/* Тихая проверка для панели: ничего не печатает, только заполняет err. */
int llm_check_quiet(const dc_config *cfg, char *err, size_t errlen)
{
    llm_status st;
    int        rc = llm_probe(cfg, &st);
    snprintf(err, errlen, "%s", st.error);
    llm_status_free(&st);
    return rc;
}

/* Список идентификаторов моделей; вызывающий освобождает элементы и массив. */
int llm_list_models(const dc_config *cfg, char ***out, char *err, size_t errlen)
{
    llm_status st;
    int        rc = llm_probe(cfg, &st);
    if (out) *out = NULL;
    if (rc == 0 && out) {
        char **ids = xcalloc((size_t)(st.nmodels > 0 ? st.nmodels : 1), sizeof *ids);
        for (int i = 0; i < st.nmodels; i++) ids[i] = xstrdup(st.models[i]);
        *out = ids;
    }
    snprintf(err, errlen, "%s", st.error);
    int n = st.nmodels;
    llm_status_free(&st);
    return n;
}

int llm_check(const dc_config *cfg, char *err, size_t errlen)
{
    llm_status st;
    int        rc = llm_probe(cfg, &st);

    printf("адрес:  %s\n", cfg->llm_url ? cfg->llm_url : "(не задан)");
    printf("модель: %s\n", cfg->llm_model && *cfg->llm_model ? cfg->llm_model : "(не задана)");
    printf("ключ:   %s\n", st.needs_key ? "есть, шлём Authorization"
                                        : (st.has_key ? "есть, но auth = none — не шлём"
                                                      : "не нужен (сервер свой)"));

    if (rc != 0) {
        if (st.needs_key && !st.has_key) {
            printf("\nключ не задан, а сервер не локальный. Что делать:\n"
                   "  api_key = $OPENAI_API_KEY    ключ есть, сервер в облаке\n"
                   "  auth = none                  сервер свой, ключ ему не нужен\n");
        }
        printf("\n%s\n", st.error);
        snprintf(err, errlen, "%s", st.error);
        llm_status_free(&st);
        return rc;
    }

    printf("спрашиваю: %s\n\nмодели на сервере:\n", st.models_url);
    for (int i = 0; i < st.nmodels; i++) {
        int ours = cfg->llm_model && *cfg->llm_model && strcmp(st.models[i], cfg->llm_model) == 0;
        printf("  %s %s\n", ours ? "->" : "  ", st.models[i]);
    }
    printf("\nготово: план можно строить%s.\n", st.needs_key ? " (ключ подставлен)" : " (без ключа)");
    err[0] = '\0';
    llm_status_free(&st);
    return 0;
}
