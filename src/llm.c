/*
 * Планирование локальной, распределённой моделью.
 *
 * Раньше здесь жил HTTP-клиент: план просили у чужого сервера — Ollama,
 * llama.cpp server или облака. Теперь просить некого: модель лежит в файле
 * GGUF, её веса разложены по узлам роли shard, а планировщик только гоняет
 * через них активацию. Ключей, адресов и стилей запроса здесь больше нет.
 *
 * Что это меняет по существу:
 *   - план строится на той же модели, которую потом разбирают по устройствам
 *     для `llm prompt`, то есть одна и та же модель и для планирования,
 *     и для работы конвейера;
 *   - память под модель не нужна планировщику вообще — она распределена;
 *   - работает без сети: ни ключа, ни адреса, ни TLS.
 */

#include "llm.h"
#include "expr.h"
#include "json.h"
#include "plan.h"
#include "shard.h"
#include "util.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Потолок контекста для планирования. Совпадает с потолком движка: больше
 * не держит ни один узел, а просить сверх бессмысленно. */
#define DC_PLAN_CTX_MAX 4096

/* ------------------------------------------------------------ промпт --- *//* ------------------------------------------------------------ промпт --- */

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
/* ------------------------------------------- запрос к локальной модели --- */

/* Промпт целиком: системный, цель и — при починке — прошлый ответ с причиной
 * отказа. Модели нужны и её собственный прошлый ответ, и точная причина:
 * просить «сделай лучше» бесполезно, просить «вот это исправь» работает. */
static char *build_prompt(const dc_config *cfg, const char *goal, const char *rejected,
                          const char *reason)
{
    strbuf b;
    sb_init(&b);
    sb_add(&b, llm_system_prompt(cfg));
    sb_add(&b, "\n\nЦЕЛЬ: ");
    sb_add(&b, goal);
    if (rejected) {
        sb_add(&b, "\n\nТВОЙ ПРЕДЫДУЩИЙ ОТВЕТ:\n");
        sb_add(&b, rejected);
        sb_add(&b, "\n\n");
        sb_add(&b, reason ? reason : "");
    }
    return b.p;
}


static char *llm_ask(dc_infer *in, const dc_config *cfg, const char *goal,
                     const char *rejected, const char *reason, char *err, size_t errlen)
{
    char *prompt = build_prompt(cfg, goal, rejected, reason);

    int max_new = cfg->max_new > 0 ? cfg->max_new : 512;
    log_info("локальная модель: %s, узлов %d, стадий %d, контекст %d токенов",
             dc_infer_arch(in), dc_infer_n_nodes(in), dc_infer_n_stages(in),
             dc_infer_n_ctx(in));

    long  t0   = now_ms();
    char *text = dc_infer_generate(in, prompt, max_new, err, errlen);
    free(prompt);
    if (!text) return NULL;

    log_info("модель ответила за %ld мс, контекста занято %d токенов", now_ms() - t0,
             dc_infer_used(in));

    char *clean = sanitize_json_text(text);
    if (!clean) {
        snprintf(err, errlen,
                 "в ответе модели нет JSON-объекта с планом; первые 200 символов: %.200s",
                 text);
        free(text);
        return NULL;
    }
    log_debug("план от модели: %s", clean);
    free(text);
    return clean;
}

/* Открыть конвейер, подобрав контекст под промпт.
 *
 * Размер промпта считается по токенизатору модели, а не «на глаз». Оценка
 * «три знака на токен» верна для модели со словарём в сотни тысяч слов, но у
 * модели со словарём в 256 слов каждый знак — отдельный токен, и такая оценка
 * заказала бы у узлов в семь раз меньше контекста, чем нужно. Потом это
 * выглядит как «узел не вмещает ни одного слоя», хотя дело в промпте.
 *
 * Если под такой контекст узлов не хватает, пробуем меньше — но не меньше
 * самого промпта: обрезанная инструкция хуже явного отказа, модель по
 * обрезанному правилу ответит не на то. */
static dc_infer *open_infer(const dc_config *cfg, const char *goal, char *err, size_t errlen)
{
    const char *model = cfg->model_file;
    if (!model || !*model) {
        snprintf(err, errlen,
                 "не задана модель: укажите model_file в секции [llm] — путь к GGUF-файлу");
        return NULL;
    }

    int   max_new = cfg->max_new > 0 ? cfg->max_new : 512;
    char *prompt  = build_prompt(cfg, goal, NULL, NULL);

    int  ntok = 0;
    char terr[512] = {0};
    if (llm_model_tokens(model, prompt, &ntok, terr, sizeof terr) != 0) {
        snprintf(err, errlen, "%s", terr);
        free(prompt);
        return NULL;
    }
    free(prompt);

    /* запас на ответ и на то, что позиция в кэше нумеруется с нуля */
    int need = ntok + max_new + 8;
    log_info("промпт %d токенов, ответ до %d, нужно контекста %d", ntok, max_new, need);

    if (cfg->n_ctx > 0) {
        /* пользователь задал размер сам: уважаем и не подгоняем молча */
        if (need > cfg->n_ctx) {
            snprintf(err, errlen,
                     "контекста %d токенов не хватит: промпт занимает %d, ответ нужен "
                     "до %d. Поднимите n_ctx в секции [llm] — но каждому узлу тогда "
                     "понадобится больше памяти под KV-кэш",
                     cfg->n_ctx, ntok, max_new);
            return NULL;
        }
        need = cfg->n_ctx;
    } else if (need > DC_PLAN_CTX_MAX) {
        snprintf(err, errlen,
                 "промпт и ответ требуют %d токенов контекста, а больше %d узлы "
                 "не держат: сократите системный промпт или возьмите модель "
                 "с более крупным словарём",
                 need, DC_PLAN_CTX_MAX);
        return NULL;
    }

    char err2[512] = {0};
    for (int i = 0; i < 2; i++) {
        /* второй проход — с меньшим контекстом, если узлы не тянут нужный */
        int n_ctx = need;
        if (i) n_ctx = need / 2;
        if (n_ctx < ntok + 8) continue;   /* промпт не влезет — не мучаемся зря */

        dc_infer      *in = NULL;
        dc_infer_opts  o  = { .n_ctx = n_ctx, .verbose = 0 };
        /* конфигурация меняется по ходу: опрос узлов заполняет их возможности,
         * и раскладка смотрит именно на них */
        if (dc_infer_open((dc_config *)cfg, model, &o, &in, err2, sizeof err2) == 0) {
            if (i) log_info("узлы не тянут контекст %d, беру %d", need, n_ctx);
            return in;
        }
        log_debug("раскладка с контекстом %d не вышла: %s", n_ctx, err2);
    }

    snprintf(err, errlen, "модель не разложилась по узлам: %s", err2);
    return NULL;
}

/* ----------------------------------------------------------------- чат --- */
/*
 * Та же модель и то же разложение, но работает человек, а не план. Поэтому
 * системный промпт здесь другой, а контекста нужно больше: разговор длиннее
 * одной задачи, и модель должна видеть предыдущие реплики.
 */

const char *llm_chat_system(const dc_config *cfg)
{
    if (cfg->chat_system && *cfg->chat_system) return cfg->chat_system;
    return "Ты — помощник. Отвечай по-русски, кратко и по делу. "
           "Если вопрос вычислимый, считай прямо в ответе.";
}

/* Промпт в формате ChatML — так его понимают Qwen2 и Llama-3.
 * history уже готовые реплики, каждая «роль» + «текст» + перевод строки. */
char *llm_chat_prompt(const char *system, const char *history, const char *message)
{
    strbuf b;
    sb_init(&b);
    sb_add(&b, "<|im_start|>system\n");
    sb_add(&b, system && *system ? system : "");
    sb_add(&b, "<|im_end|>\n");
    if (history && *history) {
        sb_add(&b, history);
    }
    sb_add(&b, "<|im_start|>user\n");
    sb_add(&b, message && message[0] ? message : "");
    sb_add(&b, "<|im_end|>\n<|im_start|>assistant\n");
    return b.p;
}

void llm_chat_history_add(strbuf *hist, const char *role, const char *text)
{
    sb_add(hist, "<|im_start|>");
    sb_add(hist, role);
    sb_add(hist, "\n");
    sb_add(hist, text && text[0] ? text : "");
    sb_add(hist, "<|im_end|>\n");
}

/* Открывает конвейер под чат. Отличие от планирования в том, сколько места
 * заказывается узлам: там хватает системного промпта и одной задачи, здесь
 * нужны ещё история разговора и длинный ответ. */
dc_infer *llm_open_chat(dc_config *cfg, int max_new, char *err, size_t errlen)
{
    const char *model = cfg->model_file;
    if (!model || !*model) {
        snprintf(err, errlen,
                 "не задана модель: укажите model_file в секции [llm] — путь к GGUF-файлу");
        return NULL;
    }
    if (max_new <= 0) max_new = cfg->max_new > 0 ? cfg->max_new : 512;

    /* Прикидка объёма: системный промпт плюс история плюс ответ. Считаем по
     * токенизатору, а не «три знака на токен» — у модели со словарём в 256
     * слов каждый знак это токен, и оценка врала бы в семь раз. */
    char *sys   = xstrdup(llm_chat_system(cfg));
    char *probe = llm_chat_prompt(sys, "", "");
    int   ntok  = 0;
    char  terr[512] = {0};
    if (llm_model_tokens(model, probe, &ntok, terr, sizeof terr) != 0) {
        snprintf(err, errlen, "%s", terr);
        free(sys); free(probe);
        return NULL;
    }
    free(probe);

    int history_msgs = cfg->chat_history > 0 ? cfg->chat_history : 8;
    /* на каждую сохранённую реплику закладываем 200 токенов — это с запасом
     * покрывает обычую реплику и не заказывает узлам лишней памяти */
    int need = ntok + history_msgs * 200 + max_new + 8;
    log_info("чат: промпт %d токенов, история %d реплик, ответ до %d, нужно контекста %d",
             ntok, history_msgs, max_new, need);

    if (need > DC_PLAN_CTX_MAX) need = DC_PLAN_CTX_MAX;
    if (cfg->n_ctx > 0) need = cfg->n_ctx;

    char err2[512] = {0};
    for (int i = 0; i < 2; i++) {
        int n_ctx = need;
        if (i) n_ctx = need / 2;
        if (n_ctx < ntok + 8) continue;
        dc_infer      *in = NULL;
        dc_infer_opts  o  = { .n_ctx = n_ctx, .verbose = 0 };
        if (dc_infer_open(cfg, model, &o, &in, err2, sizeof err2) == 0) {
            free(sys);
            if (i) log_info("узлы не тянут контекст %d, беру %d", need, n_ctx);
            return in;
        }
        log_debug("разкладка с контекстом %d не вышла: %s", n_ctx, err2);
    }
    free(sys);
    snprintf(err, errlen, "модель не разложилась по узлам: %s", err2);
    return NULL;
}

char *llm_make_plan(const dc_config *cfg, const char *goal, int use_llm, int allow_fallback,
                    char *err, size_t errlen)
{
    if (!use_llm) {
        log_info("режим --no-llm: строю план эвристикой");
        return llm_heuristic_plan(cfg, goal);
    }

    if (!cfg->model_file || !*cfg->model_file) {
        if (!allow_fallback) {
            snprintf(err, errlen,
                     "не задана модель. Задайте в конфигурации "
                     "[llm] model_file = путь/к/model.gguf и узлы роли %s — "
                     "без них план строит только эвристика",
                     ROLE_SHARD);
            return NULL;
        }
        log_info("модель не задана — использую эвристический планировщик");
        return llm_heuristic_plan(cfg, goal);
    }

    dc_infer *in = open_infer(cfg, goal, err, errlen);
    if (!in) {
        if (allow_fallback) {
            log_info("модель недоступна (%s) — эвристический планировщик", err);
            return llm_heuristic_plan(cfg, goal);
        }
        return NULL;
    }

    /*
     * Модель ошибается не только на сети: чаще план приходит негодным по
     * существу — цикл в зависимостях, роль не из списка, обрезанный ответ.
     * Молча откатываться на эвристику тут нельзя: пользователь просил
     * модель и получил бы не то, что просил, не заметив. Поэтому показываем
     * модели её собственный ответ и причину отказа, и она пробует исправить.
     * Эвристика остаётся последним рубежом.
     */
    char *rejected = NULL;
    char  reason[1024] = "";
    char  lasterr[1280] = "";

    for (int attempt = 1; attempt <= DC_LLM_TRIES; attempt++) {
        char aerr[1024] = {0};
        char *json = llm_ask(in, cfg, goal, rejected, rejected ? reason : NULL, aerr, sizeof aerr);
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
            dc_infer_free(in);
            return json;
        }

        log_error("план от модели отклонён: %s", perr);
        free(rejected);
        rejected = json;                    /* теперь это «прошлый ответ» */
        char *rp = repair_prompt(perr);
        snprintf(reason, sizeof reason, "%s", rp ? rp : "");
        free(rp);
        snprintf(lasterr, sizeof lasterr, "план от модели отклонён: %s", perr);

        if (attempt < DC_LLM_TRIES) log_info("прошу модель исправить план (попытка %d)", attempt + 1);
    }

    free(rejected);
    dc_infer_free(in);
    snprintf(err, errlen, "%s", lasterr[0] ? lasterr : "модель не вернула пригодный план");
    if (!allow_fallback) return NULL;
    log_info("перехожу на эвристический планировщик");
    return llm_heuristic_plan(cfg, goal);
}

/* -------------------------------------------------- проверка модели --- */

int llm_local_status(const dc_config *cfg, llm_status *st)
{
    memset(st, 0, sizeof *st);
    snprintf(st->model, sizeof st->model, "%s", cfg->model_file ? cfg->model_file : "");

    if (!cfg->model_file || !*cfg->model_file) {
        snprintf(st->note, sizeof st->note,
                 "не задан model_file в секции [llm] — план строится эвристикой");
        return st->ready = 0;
    }

    /* раскладка без заливки: спрашиваются только PING, веса не трогаем */
    dc_infer      *in = NULL;
    dc_infer_opts  o  = { .n_ctx = cfg->n_ctx, .verbose = 0 };
    char err[512] = {0};
    if (dc_infer_layout((dc_config *)cfg, cfg->model_file, &o, &in, err, sizeof err) != 0) {
        snprintf(st->note, sizeof st->note, "%s", err);
        return st->ready = 0;
    }

    snprintf(st->arch, sizeof st->arch, "%s", dc_infer_arch(in));
    st->n_layer  = dc_infer_n_layer(in);
    st->n_embd   = dc_infer_n_embd(in);
    st->n_vocab  = dc_infer_n_vocab(in);
    st->n_ctx    = dc_infer_n_ctx(in);
    st->n_nodes  = dc_infer_n_nodes(in);
    st->n_stages = dc_infer_n_stages(in);
    st->bytes    = dc_infer_model_bytes(in);
    dc_infer_free(in);
    return st->ready = 1;
}

int llm_check_quiet(const dc_config *cfg, char *err, size_t errlen)
{
    llm_status st;
    int        ready = llm_local_status(cfg, &st);
    snprintf(err, errlen, "%s", ready ? "" : st.note);
    return ready ? 0 : 3;
}

int llm_check(const dc_config *cfg, char *err, size_t errlen)
{
    const char *model = cfg->model_file;
    if (!model || !*model) {
        printf("модель: (не задана)\nузлы:   %s\n\n", ROLE_SHARD);
        printf("готово: нет — не задан model_file в секции [llm]\n");
        printf("\nЧто делать:\n"
               "  [llm] model_file = путь/к/model.gguf   файл модели в формате GGUF\n"
               "  [role %s] daemon = host:port             минимум три узла:\n"
               "                                             вход, слои и выход\n",
               ROLE_SHARD);
        snprintf(err, errlen, "не задан model_file в секции [llm]");
        return 3;
    }

    printf("модель: %s\nузлы:   %s\n", model, ROLE_SHARD);

    /* Раскладка печатается здесь же, а не только в `llm check`: человек,
     * который спрашивает «готов ли планировщик», хочет видеть, на что модель
     * разбилась и влезает ли она в парк. */
    dc_infer      *in = NULL;
    dc_infer_opts  o  = { .n_ctx = cfg->n_ctx, .verbose = 0 };
    char e2[512] = {0};
    if (dc_infer_layout((dc_config *)cfg, model, &o, &in, e2, sizeof e2) != 0) {
        printf("\nготово: нет — %s\n", e2);
        snprintf(err, errlen, "%s", e2);
        return 3;
    }

    printf("\nархитектура:    %s\n", dc_infer_arch(in));
    printf("слоёв:          %d\n", dc_infer_n_layer(in));
    printf("размер вектора: %d\n", dc_infer_n_embd(in));
    printf("словарь:        %d\n", dc_infer_n_vocab(in));
    printf("контекст:       %d токенов\n", dc_infer_n_ctx(in));
    printf("размер файла:   %zu байт\n", dc_infer_model_bytes(in));
    dc_infer_print_layout(in);
    printf("готово: план можно строить локально — ключи и внешний сервер не нужны.\n");
    dc_infer_free(in);
    err[0] = '\0';
    return 0;
}
