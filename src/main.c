#include "conf.h"
#include "json.h"
#include "llm.h"
#include "plan.h"
#include "sched.h"
#include "shard.h"
#include "tty.h"
#include "udp.h"
#include "util.h"
#include "web.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <getopt.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static dc_web *g_web = NULL;

static void on_sigint(int sig)
{
    (void)sig;
    if (g_web) web_stop(g_web);
}

/*
 * Слушает широковещательные объявления узлов и печатает найденные.
 * Нужен, чтобы не вписывать каждый телефон и плату руками: узел сам
 * сообщает, какие роли умеет и сколько байт payload принимает.
 */

/* Уже увиденный узел. Адрес источника может отличаться от адреса службы
 * (объявление уходит с эфемерного сокета), поэтому опознаём узел по
 * имени и рабочему порту, а не по адресу пакета. */
typedef struct {
    char addr[64];
    char node[48];
    int  port;
    int  loopback;
} disco_seen;

/* --------------------------------------------------- пул рабочих ---- */
/*
 * Рабочий роль не объявляет: объявление несёт только его мощь. Кем он будет
 * и сколько сделает, решает распределитель на каждой задаче — а значит, роль
 * не обязана быть прописана в конфигурации заранее, рабочих можно найти самим.
 *
 * Дозаполняем пул только там, где узлов действительно не хватает: лишние две
 * секунды ожидания перед каждой командой никому не нужны. Один раз на процесс.
 */
static int pool_tried = 0;

static void fill_pool(dc_config *cfg, const char *need_role)
{
    if (pool_tried) return;
    dc_role *r = conf_role(cfg, need_role, 0);
    if (r && r->nep > 0) return;   /* роль описана в конфигурации — не мешаем */
    pool_tried = 1;

    char perr[256] = {0};
    int  n = sched_pool_fill(cfg, 2, perr, sizeof perr);
    if (n < 0) {
        log_info("рабочие не искались: %s", perr);
        return;
    }
    if (n > 0) log_info("найдено рабочих: %d", n);
    else       log_info("рабочих в сети не найдено — беру узлы из конфигурации");
}

/* Прописана ли роль, отличная от пула: если да, конфигурация осмысленная
 * и искать рабочих незачем. */
static int roles_configured(const dc_config *cfg)
{
    for (int i = 0; i < cfg->nroles; i++)
        if (cfg->roles[i].name && strcmp(cfg->roles[i].name, ROLE_POOL) != 0 &&
            cfg->roles[i].nep > 0)
            return 1;
    return 0;
}

/* ------------------------------------------------------- discover ---- */

static int discover_listen(int seconds)
{
    if (seconds < 1) seconds = 1;
    if (seconds > 300) seconds = 300;

    char err[256] = {0};
    int  fd = udp_socket_bound("0.0.0.0", DC_DISCOVERY_PORT, err, sizeof err);
    if (fd < 0) {
        log_error("%s", err);
        return 0;
    }
    int on = 1;
    setsockopt(fd, SOL_SOCKET, SO_BROADCAST, &on, sizeof on);
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);

    printf("Слушаю объявления узлов %d с (Ctrl-C — прервать)…\n", seconds);
    fflush(stdout);

    /* Узлы объявляют себя по таймеру, поэтому одного «слушания» мало: если
     * объявление прошло мимо, ждать придётся до следующего. Раз в полсекунды
     * спрашиваем сеть «кто тут?» — узел отвечает объявлением немедленно,
     * и поиск занимает столько, сколько занимает сеть. */
    int ask_fd = udp_socket_client(NULL, 0);
    /* без SO_BROADCAST отправка широковещания кончается EACCES — и молча,
     * потому что код возврата sendto мы здесь игнорируем ради простоты цикла */
    if (ask_fd >= 0) setsockopt(ask_fd, SOL_SOCKET, SO_BROADCAST, &on, sizeof on);

    char ask_json[] = "{\"role\":\"discover\"}";
    struct sockaddr_in ask_addr;
    memset(&ask_addr, 0, sizeof ask_addr);
    ask_addr.sin_family = AF_INET;
    ask_addr.sin_port = htons(DC_DISCOVERY_PORT);

    int      found = 0;
    disco_seen seen[32];
    int      nseen = 0;
    memset(seen, 0, sizeof seen);
    long     deadline = now_ms() + (long)seconds * 1000;
    long     next_ask = 0;

    while (now_ms() < deadline) {
        if (ask_fd >= 0 && now_ms() >= next_ask) {
            unsigned char *d = NULL;
            size_t         dl = 0;
            if (udp_pack(DC_MSG_PING, ask_json, &d, &dl) == 0) {
                static const unsigned long bcast[2] = { 0xFFFFFFFFu, 0x7FFFFFFFu };
                for (int i = 0; i < 2; i++) {
                    ask_addr.sin_addr.s_addr = htonl(bcast[i]);
                    if (sendto(ask_fd, d, dl, 0, (struct sockaddr *)&ask_addr,
                               sizeof ask_addr) < 0)
                        log_debug("широковещательный запрос не ушёл: %s", strerror(errno));
                }
            }
            free(d);
            next_ask = now_ms() + 500;
        }

        unsigned char      buf[2048];
        struct sockaddr_in from;
        socklen_t          flen = sizeof from;
        struct timeval     tv = { 0, 200000 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

        ssize_t r = recvfrom(fd, buf, sizeof buf, 0, (struct sockaddr *)&from, &flen);
        if (r <= 0) continue;

        dc_msg m;
        char   perr[192] = {0};
        if (udp_unpack(buf, (size_t)r, &m, perr, sizeof perr) != 0) continue;
        if (m.type != DC_MSG_HELLO) continue;

        char json[2049];
        size_t len = m.body_len < sizeof json - 1 ? m.body_len : sizeof json - 1;
        memcpy(json, m.body, len);
        json[len] = '\0';

        char     jerr[192] = {0};
        js_doc  *d = js_parse(json, jerr, sizeof jerr);
        if (!d) continue;
        const js_value *root = js_root(d);
        const char *addr = inet_ntoa(from.sin_addr);
        const char *role = js_get_str(root, "role");
        const char *node = js_get_str(root, "node");
        const char *arch = js_get_str(root, "arch");
        double      mb = js_num(js_get(root, "max_body"), 0);
        /* порт берём из объявления: источник HELLO — эфемерный сокет */
        double      sport = js_num(js_get(root, "port"), ntohs(from.sin_port));
        const char *nm = node ? node : "?";
        int         p = (int)sport;

        /* Ключ — имя узла и его рабочий порт, а не адрес источника:
         * один и тот же узел мы слышим и по 127.0.0.1, и по адресу
         * сетевого интерфейса (мы спрашиваем по обоим широковещаниям),
         * и каждые полсекунды, потому что сами просим «кто тут?».
         * В список попадает одна строка на узел, и адрес из неё — тот,
         * по которому до него можно доехать, то есть не loopback. */
        int loop = from.sin_addr.s_addr >> 24 == 127;
        int slot = -1;
        for (int i = 0; i < nseen; i++)
            if (seen[i].port == p && strcmp(seen[i].node, nm) == 0) { slot = i; break; }

        if (slot >= 0) {
            if (!(loop && !seen[slot].loopback)) {   /* более полезный адрес уже есть */
                js_doc_free(d);
                continue;
            }
            snprintf(seen[slot].addr, sizeof seen[slot].addr, "%s", addr);
            seen[slot].loopback = loop;
            js_doc_free(d);
            continue;   /* адрес уточнили, печатать второй раз нечего */
        }
        if (nseen < (int)(sizeof seen / sizeof seen[0])) {
            snprintf(seen[nseen].addr, sizeof seen[nseen].addr, "%s", addr);
            snprintf(seen[nseen].node, sizeof seen[nseen].node, "%s", nm);
            seen[nseen].port = p;
            seen[nseen].loopback = loop;
            nseen++;
        }
        found++;

        printf("  %-16s:%-6d %-14s %-28s лимит %6.0f байт  %s\n", addr, p, nm,
               role ? role : "?", mb, arch ? arch : "");
        fflush(stdout);
        js_doc_free(d);
    }
    close(fd);
    if (ask_fd >= 0) close(ask_fd);

    printf("\nнайдено объявлений: %d\n", found);
    if (found) {
        printf("\nэти рабочие и составят пул: роль им назначу сам, по одной на задачу,\n"
               "а сколько работы достанется — по их памяти.\n"
               "Ничего в конфигурацию вписывать не нужно.\n");
        printf("Задать роль узлу намертво можно секцией [role compute]\n"
               "  daemon = 192.168.1.50:9101\n");
    }
    return found;
}

static void usage(FILE *out)
{
    fprintf(out,
"dcctl — LLM-планировщик распределённых вычислений\n"
"\n"
"Использование:\n"
"  dcctl [опции] <команда> [цель]\n"
"\n"
"Команды:\n"
"  init [ПУТЬ]         создать образец конфигурации (по умолчанию ./distcomp.conf)\n"
"  roles              показать, какие узлы обслуживают какие роли\n"
    "  runs               прошлые запуски из outdir (дата, итог, шаги)\n"
    "  replay МЕТКА       повторить сохранённый план (или путь к файлу)\n"
"  check              проверить узлы (UDP PING/PONG)\n"
"  discover [СЕК]     найти узлы, объявляющие себя по широковещанию\n"
"  plan \"<цель>\"      построить план распределения и показать его\n"
    "  shard gen|info|deploy|run   модель, разложенная по устройствам по памяти\n"
    "  llm                   проверить, что LLM настроена и отвечает\n"
"  run  \"<цель>\"      построить план и выполнить его\n"
"  run  -f план.json   выполнить уже готовый план\n"
"  launch             показать команды запуска dcd для всех узлов из конфигурации\n"
"  web  [--port 8080]  веб-панель управления (открыть в браузере)\n"
"\n"
"Опции:\n"
"  -c, --config ФАЙЛ     конфигурация (по умолчанию ./distcomp.conf, ~/.distcomp.conf)\n"
"      --role РОЛЬ=URL    закрепить узел за ролью (можно повторять); без этого\n"
            "                        распределитель сам выбирает из пула рабочих\n"
"  -f, --plan-file ФАЙЛ  исполнить готовый план\n"
"  -j, --concurrency N   одновременно выполняемых шагов\n"
"      --retries N       повторов при ошибке узла\n"
"      --timeout МС      таймаут одного узла\n"
"      --no-llm          не обращаться к LLM, построить план эвристикой\n"
"      --strict          не откатываться на эвристику при ошибке LLM\n"
"  -o, --out ДИР         куда сохранять plan.json / result.json\n"
"      --host АДРЕС      адрес веб-панели (по умолчанию 0.0.0.0, только для web)\n"
"      --port ПОРТ       порт веб-панели (по умолчанию 8080, только для web)\n"
"  -v, --verbose         подробный лог\n"
"      --color УРОВЕНЬ   цвет вывода: auto | always | never (auto — терминал без NO_COLOR)\n"
"  -V, --version         версия\n"
"  -h, --help            эта справка\n");
}

static int ensure_dir(const char *path)
{
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0775);
            *p = '/';
        }
    }
    mkdir(tmp, 0775);
    struct stat st;
    return stat(tmp, &st) == 0 && S_ISDIR(st.st_mode);
}

/* Каталог запусков. Никогда не берём его как есть: пустое или отсутствующее
 * значение превратилось бы в «(null)/метка» — каталог с именем «(null)» в
 * рабочем каталоге. Проще подставить значение по умолчанию. */
static const char *outdir_or_default(const char *outdir)
{
    return (outdir && *outdir) ? outdir : "runs";
}

/* Метка запуска — это имя каталога в outdir. Два запуска в одну секунду (или
 * два процесса рядом) не должны затирать друг друга, поэтому занятая метка
 * получает счётчик. */
static void run_id_now(char *buf, size_t n, const char *outdir)
{
    outdir = outdir_or_default(outdir);
    time_t    t = time(NULL);
    struct tm tmv;
    localtime_r(&t, &tmv);

    char base[32];
    strftime(base, sizeof base, "%Y%m%d-%H%M%S", &tmv);
    for (int k = 0; k < 100; k++) {
        char cand[40];
        if (k == 0) snprintf(cand, sizeof cand, "%s", base);
        else        snprintf(cand, sizeof cand, "%s-%02d", base, k);
        if (strlen(cand) + 1 > n) break;      /* буфер вызывающего мал */
        memcpy(buf, cand, strlen(cand) + 1);

        char dir[1200];
        snprintf(dir, sizeof dir, "%s/%s", outdir, buf);
        struct stat st;
        if (stat(dir, &st) != 0) return;   /* каталога нет — метка свободна */
    }
    snprintf(buf, n, "%s", base);
}

static void write_file(const char *dir, const char *name, const char *content);
static int  do_run(dc_config *cfg, const char *goal, const char *plan_file, int execute,
                   int no_llm, int strict, char *err, size_t errlen);

/* Каталоги запусков в outdir: метка — это имя каталога, метка времени
 * упорядочена по строке, поэтому сортировка обычная строковая. */
static int read_runs(const char *outdir, char (*ids)[64], int max)
{
    DIR *d = opendir(outdir_or_default(outdir));
    if (!d) return -1;
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) && n < max) {
        if (e->d_name[0] == '.') continue;
        struct stat st;
        char        p[1200];
        snprintf(p, sizeof p, "%s/%s", outdir_or_default(outdir), e->d_name);
        if (stat(p, &st) != 0 || !S_ISDIR(st.st_mode)) continue;
        /* длинные имена каталогов обрезаем, но метка должна влезть в буфер */
        snprintf(ids[n], 64, "%.63s", e->d_name);
        n++;
    }
    closedir(d);
    for (int i = 1; i < n; i++) {   /* вставка: метки идут в хронологическом порядке */
        char tmp[64];
        memcpy(tmp, ids[i], 64);
        int j = i - 1;
        while (j >= 0 && strcmp(ids[j], tmp) > 0) {
            memcpy(ids[j + 1], ids[j], 64);
            j--;
        }
        memcpy(ids[j + 1], tmp, 64);
    }
    return n;
}

/* Короткая сводка по result.json: «выполнено/сломано», число шагов. */
static void run_one_line(const char *outdir, const char *id)
{
    char  path[1200];
    snprintf(path, sizeof path, "%s/%s/result.json", outdir, id);
    FILE *f = fopen(path, "r");
    if (!f) {
        printf("  %-20s %s\n", id, "план есть, результата нет");
        return;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = xmalloc((size_t)sz + 1);
    size_t got = fread(buf, 1, (size_t)sz, f);
    buf[got] = '\0';
    fclose(f);

    char     perr[256] = {0};
    js_doc  *d = js_parse(buf, perr, sizeof perr);
    free(buf);
    if (!d) {
        printf("  %-20s %s\n", id, "result.json не читается");
        return;
    }
    const js_value *root = js_root(d);
    const js_value *okv = js_get(root, "ok");
    const js_value *st  = js_get(root, "steps");
    int            done = 0, total = 0;
    if (st && st->type == JS_ARR) {
        total = (int)st->len;
        for (size_t i = 0; i < st->len; i++) {
            const char *sname = js_get_str(js_at(st, i), "state");
            if (sname && strcmp(sname, "готово") == 0) done++;
        }
    }
    if (okv && okv->type == JS_BOOL && okv->bval) {
        printf("  ");
        TC(stdout, C_GREEN, "%-20s", id);
    } else {
        printf("  ");
        TC(stdout, C_RED, "%-20s", id);
    }
    printf(" шагов %d из %d", done, total);
    const char *goal = js_get_str(root, "goal");
    if (goal && *goal) printf("   %s", goal);
    putchar('\n');
    js_doc_free(d);
}

static int cmd_runs(dc_config *cfg)
{
    char ids[64][64];
    int  n = read_runs(cfg->outdir, ids, 64);
    if (n < 0) {
        log_error("каталога запусков нет: %s (создастся при первом plan/run)", cfg->outdir);
        return 3;
    }
    if (n == 0) {
        printf("Запусков пока нет в %s\n", cfg->outdir);
        return 0;
    }
    printf("Запуски в %s (новые сверху):\n\n", cfg->outdir);
    for (int i = n - 1; i >= 0; i--) run_one_line(cfg->outdir, ids[i]);
    printf("\nПовторить: dcctl replay %s\n", ids[n - 1]);
    return 0;
}

/* replay: метка из outdir либо готовый путь к plan.json */
static int cmd_replay(dc_config *cfg, const char *what, int no_llm, int strict, char *err,
                      size_t errlen)
{
    char path[1200];
    if (strchr(what, '/') || strstr(what, ".json"))
        snprintf(path, sizeof path, "%s", what);
    else
        snprintf(path, sizeof path, "%s/%s/plan.json",
                 outdir_or_default(cfg->outdir), what);

    /* отличаем «нет такого запуска» от «есть, но сломано» */
    FILE *f = fopen(path, "r");
    if (!f) {
        /* путь режем сам: он и так длинный, а сообщение читают глазами */
        snprintf(err, errlen,
                 "нет плана для «%.80s» (%.240s) — посмотрите список: dcctl runs", what, path);
        return 3;
    }
    fclose(f);
    return do_run(cfg, NULL, path, 1, no_llm, strict, err, errlen);
}

/* Собрать план и выполнить его. goal — на естественном языке (NULL, если
 * берём готовый план из plan_file). Единая точка для run и replay: повтор
 * отличается только тем, что план не строится заново. */
static int do_run(dc_config *cfg, const char *goal, const char *plan_file, int execute,
                  int no_llm, int strict, char *err, size_t errlen)
{
    /* Выполняем всегда, когда это команда run: готовый план из -f или из
     * replay тоже должен уйти на узлы. План без выполнения — это `plan`. */
    int    running = execute;
    char  *plan_json = NULL;

    if (plan_file) {
        FILE *f = fopen(plan_file, "r");
        if (!f) {
            snprintf(err, errlen, "не удалось открыть план %s", plan_file);
            return 3;
        }
        strbuf b;
        sb_init(&b);
        char   chunk[4096];
        size_t got;
        while ((got = fread(chunk, 1, sizeof chunk, f)) > 0) sb_addn(&b, chunk, got);
        fclose(f);
        plan_json = b.p;
        log_info("беру план из файла %s", plan_file);
    } else {
        if (!goal || !*goal) {
            snprintf(err, errlen, "нужна цель: dcctl run \"описание задачи\"");
            return 2;
        }
        log_info("цель: %s", goal);
        /* Возможности узлов узнаём до обращения к модели: в промпте стоят
         * реальные лимиты тела, и модель не режет задачу под маленькую плату */
        if (!no_llm) sched_learn_caps(cfg);
        plan_json = llm_make_plan(cfg, goal, !no_llm, !strict, err, errlen);
        if (!plan_json) return 4;
    }

    plan *p = plan_from_json(plan_json, cfg, err, errlen);
    if (!p) {
        free(plan_json);
        return 4;
    }
    log_debug("сырой план: %s", plan_json);

    printf("\n");
    sched_print_plan(p, cfg);

    char rid[32];
    const char *outdir = outdir_or_default(cfg->outdir);
    run_id_now(rid, sizeof rid, outdir);

    char rundir[1024];
    snprintf(rundir, sizeof rundir, "%s/%s", outdir, rid);
    ensure_dir(rundir);

    char *pretty = plan_to_json(p);
    write_file(rundir, "plan.json", pretty);
    free(pretty);

    int rc = 0;
    int ok = 1;
    if (running) {
        rc = sched_run(p, cfg);
        char *res = plan_result_json(p, rid);
        write_file(rundir, "result.json", res);
        free(res);
        printf("\n");

        for (int i = 0; i < p->n; i++)
            if (p->steps[i].state != ST_DONE) ok = 0;

        printf("Итог:\n");
        for (int i = 0; i < p->n; i++) {
            const plan_step *s = &p->steps[i];
            printf("  ");
            TC(stdout, state_color(s->state), "%s", state_icon(s->state));
            printf(" %-4s %-9s %-28s ", s->id, s->role, s->ep ? s->ep : "-");
            if (s->output && *s->output) TC(stdout, C_BOLD, "%s", s->output);
            else if (s->error && *s->error) TC(stdout, C_RED, "%s", s->error);
            putchar('\n');
        }
        putchar('\n');
        if (ok) TC(stdout, C_GREEN, "Все шаги выполнены.");
        else    TC(stdout, C_RED, "Есть незавершённые шаги — см. result.json.");
        printf("\nКаталог запуска: %s\n", rundir);
    }

    plan_free(p);
    free(plan_json);
    return running && !ok ? 1 : rc;    /* есть незавершённые шаги — код 1 */
}

static void write_file(const char *dir, const char *name, const char *content)
{
    char path[1200];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "w");
    if (!f) {
        log_error("не удалось записать %s", path);
        return;
    }
    fputs(content, f);
    fclose(f);
    printf("  сохранено: %s\n", path);
}

int main(int argc, char **argv)
{
    signal(SIGPIPE, SIG_IGN);
    setvbuf(stdout, NULL, _IOLBF, 0);
    tty_init(-1);   /* --color может переопределить это позже */

    const char  *config_path = NULL;
    const char  *plan_file = NULL;
    const char  *goal = NULL;
    char        *goal_buf = NULL;
    const char  *cmd = NULL;
    int          no_llm = 0, strict = 0;
    int          concurrency = -1, retries = -1;
    long         timeout = -1;
    const char  *outdir = NULL;
    const char  *web_host = "0.0.0.0";
    int          web_port = 8080;
    char        *role_overrides[16];
    int          n_overrides = 0;

    static struct option lopts[] = {
        { "config",      required_argument, 0, 'c' },
        { "role",        required_argument, 0, 'R' },
        { "plan-file",   required_argument, 0, 'f' },
        { "concurrency", required_argument, 0, 'j' },
        { "retries",     required_argument, 0, 1001 },
        { "timeout",     required_argument, 0, 1002 },
        { "no-llm",      no_argument,       0, 1003 },
        { "strict",      no_argument,       0, 1004 },
        { "out",         required_argument, 0, 'o' },
        { "port",        required_argument, 0, 'p' },
        { "host",        required_argument, 0, 1005 },
        { "verbose",     no_argument,       0, 'v' },
        { "color",       required_argument, 0, 1006 },
        { "version",     no_argument,       0, 'V' },
        { "help",        no_argument,       0, 'h' },
        { 0, 0, 0, 0 }
    };

    optind = 1;
    int c;
    /* «+» в начале optstring: разбор опций заканчивается на первом же
     * не-опции, то есть на команде. Иначе число «-0.25» из аргументов
     * `shard run` было бы принято за флаг и вызвало справку. */
    while ((c = getopt_long(argc, argv, "+c:R:f:j:o:hvp:", lopts, NULL)) != -1) {
        switch (c) {
        case 'c': config_path = optarg; break;
        case 'R':
            if (n_overrides < 16) role_overrides[n_overrides++] = optarg;
            break;
        case 'f': plan_file = optarg; break;
        case 'j': concurrency = atoi(optarg); break;
        case 'o': outdir = optarg; break;
        case 'v': g_verbose = 1; break;
        case 'V': printf("distcomp %s\n", DC_VERSION); return 0;
        case 1006:
            if (strcmp(optarg, "always") == 0)     tty_init(1);
            else if (strcmp(optarg, "never") == 0) tty_init(0);
            else if (strcmp(optarg, "auto") == 0)  tty_init(-1);
            else {
                log_error("--color ожидает auto | always | never, получено: %s", optarg);
                return 2;
            }
            break;
        case 'p': web_port = atoi(optarg); break;
        case 1005: web_host = optarg; break;
        case 'h': usage(stdout); return 0;
        case 1001: retries = atoi(optarg); break;
        case 1002: timeout = strtol(optarg, NULL, 10); break;
        case 1003: no_llm = 1; break;
        case 1004: strict = 1; break;
        default: usage(stderr); return 2;
        }
    }

    if (optind < argc) cmd = argv[optind++];
    if (optind < argc) {
        strbuf g;
        sb_init(&g);
        for (int i = optind; i < argc; i++) {
            if (i > optind) sb_addc(&g, ' ');
            sb_add(&g, argv[i]);
        }
        goal = goal_buf = g.p;
    }

    if (!cmd || strcmp(cmd, "help") == 0) {
        usage(cmd ? stdout : stderr);
        return cmd ? 0 : 2;
    }

    /* init работает без существующего конфига */
    if (strcmp(cmd, "init") == 0) {
        const char *target = goal ? goal : (outdir ? outdir : "distcomp.conf");
        int rc = conf_save_example(target);
        if (rc == 0) printf("создан образец конфигурации: %s\n", target);
        else          log_error("не удалось создать %s", target);
        return rc == 0 ? 0 : 1;
    }

    char      err[1024] = {0};
    dc_config *cfg = conf_load(config_path, err, sizeof err);
    if (!cfg) {
        if (strstr(err, "не найден")) {
            log_error("%s", err);
            log_error("создать образец: distcomp.conf (см. conf/distcomp.conf.example)");
            return 3;
        }
        log_error("ошибка конфигурации: %s", err);
        return 3;
    }

    /* переопределения от пользователя: --role verify=http://10.0.0.5:9103 */
    for (int i = 0; i < n_overrides; i++) {
        char  *copy = xstrdup(role_overrides[i]);
        char  *eq = strchr(copy, '=');
        if (!eq) {
            log_error("--role ожидает РОЛЬ=URL, получено: %s", role_overrides[i]);
            free(copy);
            conf_free(cfg);
            return 2;
        }
        *eq = '\0';
        char *role = str_trim(copy);
        char *url = str_trim(eq + 1);
        dc_role *r = conf_role(cfg, role, 1);
        if (!r || conf_role_add_ep(r, url) != 0) {
            log_error("не удалось назначить узла для роли %s", role);
            free(copy);
            conf_free(cfg);
            return 2;
        }
        log_info("роль %s назначена узлу %s (переопределение CLI)", role, url);
        free(copy);
    }

    if (concurrency > 0) cfg->concurrency = concurrency;
    if (retries >= 0) cfg->retries = retries;
    if (timeout > 0) cfg->timeout_ms = timeout;
    if (outdir) {
        free(cfg->outdir);
        cfg->outdir = xstrdup(outdir);
    }

    /* ---------------------------------------------------- discover --- */
    if (strcmp(cmd, "discover") == 0) {
        /* число секунд пришло уже разобранным как «цель»: dcctl discover 10 */
        int seconds = goal ? atoi(goal) : 5;
        int found = discover_listen(seconds);
        conf_free(cfg);
        free(goal_buf);
        return found > 0 ? 0 : 1;
    }

    /* ---------------------------------------------------- web ------- */
    if (strcmp(cmd, "web") == 0) {
        /* У веб-панели свои опции, и их привычно пишут после команды
         * («dcctl web --port 8080»). Общий разбор на этом месте уже кончен:
         * глобальные опции стоят до команды, иначе число вида -0.25 в
         * аргументах shard съедалось бы как флаг. */
        for (int i = optind - 1; i < argc; i++) {
            const char *a = argv[i];
            if ((strcmp(a, "--port") == 0 || strcmp(a, "-p") == 0) && i + 1 < argc)
                web_port = atoi(argv[++i]);
            else if (strcmp(a, "--host") == 0 && i + 1 < argc)
                web_host = argv[++i];
        }
        fill_pool(cfg, ROLE_SHARD);
        char     werr[512] = {0};
        dc_web  *w = web_start(cfg, web_host, web_port, "/api", no_llm, werr, sizeof werr);
        if (!w) {
            log_error("%s", werr);
            conf_free(cfg);
            free(goal_buf);
            return 3;
        }
        g_web = w;
        signal(SIGINT, on_sigint);
        signal(SIGTERM, on_sigint);

        printf("Панель управления: http://%s:%d\n",
               strcmp(web_host, "0.0.0.0") == 0 ? "localhost" : web_host, web_port);
        printf("остановить: Ctrl-C\n\n");
        web_serve(w);
        /* web_free здесь не зовём: потоки соединений могут ещё дочитывать
         * сокет, а процесс сейчас завершится — освобождать нечего */
        conf_free(cfg);
        free(goal_buf);
        return 0;
    }

    /* ---------------------------------------------------- llm ------ */
    /* «dcctl llm» без подкоманды — проверка локальной модели (то же, о чём
     * планировщик думает перед постройкой плана). С подкомандой
     * info/deploy/prompt/check — работа с моделью, разложенной по узлам. */
    if (strcmp(cmd, "llm") == 0) {
        fill_pool(cfg, ROLE_SHARD);
        int rc;
        if (argc - optind > 0) {
            rc = shard_cmd(cfg, argc - optind, argv + optind, err, sizeof err);
        } else {
            rc = llm_check(cfg, err, sizeof err);
        }
        if (rc != 0 && err[0]) log_error("%s", err);
        conf_free(cfg);
        free(goal_buf);
        return rc;
    }

    /* ---------------------------------------------------- shard ----- */
    if (strcmp(cmd, "shard") == 0) {
        /* optind уже стоит за словом «shard»: дальше идут подкоманда,
         * файл модели и числа входа */
        int rc = shard_cmd(cfg, argc - optind, argv + optind, err, sizeof err);
        if (rc != 0 && err[0]) log_error("%s", err);
        conf_free(cfg);
        free(goal_buf);
        return rc;
    }

    /* ---------------------------------------------------- runs ------ */
    if (strcmp(cmd, "runs") == 0) {
        int rc = cmd_runs(cfg);
        conf_free(cfg);
        free(goal_buf);
        return rc;
    }

    if (strcmp(cmd, "replay") == 0) {
        if (!goal || !*goal) {
            log_error("нужна метка запуска: dcctl replay 20250101-120000");
            conf_free(cfg);
            free(goal_buf);
            return 2;
        }
        if (!conf_has_daemons(cfg)) fill_pool(cfg, ROLE_COMPUTE);
        if (!conf_has_daemons(cfg)) {
            log_error("ни одной роли не обеспечено узлом — выполнять нечего.");
            conf_free(cfg);
            free(goal_buf);
            return 3;
        }
        int rc = cmd_replay(cfg, goal, no_llm, strict, err, sizeof err);
        if (rc != 0 && err[0]) log_error("%s", err);
        conf_free(cfg);
        free(goal_buf);
        return rc;
    }

    /* ---------------------------------------------------- roles ----- */
    if (strcmp(cmd, "roles") == 0) {
        printf("Назначение ролей (выбираете вы):\n\n");
        if (cfg->nroles == 0) {
            printf("  ни одной роли не настроено. Добавьте секции [role compute] и т.п.\n");
            conf_free(cfg);
            free(goal_buf);
            return 3;
        }
        for (int i = 0; i < cfg->nroles; i++) {
            dc_role *r = &cfg->roles[i];
            printf("  ");
            TC(stdout, C_BOLD, "%-9s", r->name);
            if (r->nep == 0) {
                printf("— узлы не назначены\n");
                continue;
            }
            printf("\n");
            for (int j = 0; j < r->nep; j++) {
                const dc_endpoint *e = &r->eps[j];
                printf("             %d. ", j + 1);
                TC(stdout, C_CYAN, "%-24s", e->url);
                if (e->cap.known && e->cap.max_body > 0) printf(" тело до %d Б", e->cap.max_body);
                if (e->cap.free_mem > 0) printf(", под модель %ld Б", e->cap.free_mem);
                printf("  (");
                TC(stdout, C_GREEN, "%ld выполнено", e->completed);
                if (e->failed) {
                    printf(", ");
                    TC(stdout, C_RED, "%ld с ошибкой", e->failed);
                }
                printf(")\n");
            }
        }
        printf("\nВстроенные роли: %s, %s, %s, %s\n", ROLE_COMPUTE, ROLE_VERIFY, ROLE_MEMORIZE,
               ROLE_SHARD);
        conf_free(cfg);
        free(goal_buf);
        return 0;
    }

    /* ---------------------------------------------------- launch ---- */
    if (strcmp(cmd, "launch") == 0) {
        if (!conf_has_daemons(cfg)) {
            log_error("ни одной роли не обеспечено узлом — запускать нечего.");
            conf_free(cfg);
            free(goal_buf);
            return 3;
        }
        printf("Команды запуска dcd для узлов из конфигурации:\n\n");
        for (int i = 0; i < cfg->nroles; i++) {
            dc_role *r = &cfg->roles[i];
            for (int j = 0; j < r->nep; j++) {
                dc_endpoint *e = &r->eps[j];
                char *cmd_str = conf_launch_cmd(cfg, r->name, e);
                printf("# %s (%d)\n", r->name, j + 1);
                printf("%s\n\n", cmd_str);
                free(cmd_str);
            }
        }
        conf_free(cfg);
        free(goal_buf);
        return 0;
    }

    /* ---------------------------------------------------- check ----- */
    if (strcmp(cmd, "check") == 0) {
        if (!roles_configured(cfg)) fill_pool(cfg, ROLE_POOL);
        int rc = sched_check(cfg);
        conf_free(cfg);
        free(goal_buf);
        return rc == 0 ? 0 : 1;
    }

    if (strcmp(cmd, "plan") != 0 && strcmp(cmd, "run") != 0) {
        log_error("неизвестная команда: %s", cmd);
        usage(stderr);
        conf_free(cfg);
        free(goal_buf);
        return 2;
    }

    if (!conf_has_daemons(cfg)) fill_pool(cfg, ROLE_COMPUTE);
    if (!conf_has_daemons(cfg)) {
        log_error("ни одной роли не обеспечено узелом — распределять нечего.");
        log_error("Добавьте, например:  [role %s]\n  daemon = 10.0.0.5:9101", ROLE_COMPUTE);
        log_error("или оставьте рабочих в этой сети — найду их сам (dcctl discover).");
        conf_free(cfg);
        free(goal_buf);
        return 3;
    }

    int running = strcmp(cmd, "run") == 0;
    int rc = do_run(cfg, plan_file ? NULL : goal, plan_file, running, no_llm, strict, err,
                    sizeof err);
    if (rc != 0 && err[0]) log_error("%s", err);

    conf_free(cfg);
    free(goal_buf);
    return rc;
}

