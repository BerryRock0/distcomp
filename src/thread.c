/*
 * Пул потоков тяжёлого узла. Разбивает работу matvec по строкам выхода.
 *
 * Тонкость, из-за которой пул сделан один на процесс, а не на движок:
 * matvec — функция без состояния, всё нужное приходит аргументами. А
 * контекстов движка на узле может быть три (вход, слои, выход), и локальный
 * движок планировщика — ещё три. Свой пул у каждого означал бы, что
 * ожидающие потоки висят зря, а общий — работает как положено: считает тот,
 * кто сейчас считает.
 */
#include "thread.h"

#include <stdlib.h>
#include <string.h>

#ifdef DC_LITE

/* Микроконтроллер: ядро одно, а место под стеки потоков жалко. Всё считаем
 * сами — это ровно то поведение, что было до появления пула. */
int  tp_default(void)  { return 1; }
void tp_set(int n)      { (void)n; }
int  tp_threads(void)  { return 1; }
int  tp_run(tp_fn fn, void *arg, int n) { fn(arg, 0, n); return 0; }
int  tp_worthwhile(int n) { (void)n; return 0; }
void tp_shutdown(void)  {}

#else

#include <pthread.h>
#include <unistd.h>

#define TP_MAX_THREADS 64
/* Стек помощника: matvec кладёт на стек один блок в 256 float (1 КБ).
 * Значение по умолчанию в 8 МБ на шестнадцать потоков — это 128 МБ, которых
 * на встроенной ноде может просто не быть, хотя занято будет 16 КБ. */
#define TP_STACK_BYTES (512 * 1024)
/* Порог, ниже которого разбиение не окупается: накладные расходы на
 * синхронизацию съедают выигрыш. */
#define TP_MIN_OPS 65536

typedef struct {
    pthread_t       th[TP_MAX_THREADS];
    pthread_mutex_t mu;
    pthread_cond_t  cv_work;
    pthread_cond_t  cv_done;
    tp_fn           fn;
    void           *arg;
    int             n;
    int             nth;       /* потоков всего, включая вызывающий */
    int             helpers;   /* сколько помощников реально поднято */
    unsigned long   gen;       /* номер задания: пока он прежний, спать */
    int             busy;      /* сколько помощников ещё считают */
    int             stop;
    int             want;      /* сколько потоков просил tp_set; -1 — не задано */
    int             ready;
} tp_pool;

typedef struct {
    tp_pool *pool;
    int      id;          /* 0..helpers-1; вызывающий поток — helpers */
} tp_arg;

static tp_pool P;
static tp_arg  ARGS[TP_MAX_THREADS];

/* Границы куска: равномерно, без остатка, чтобы ни один поток не ждал. */
static void tp_slice(int id, int nth, int n, int *from, int *to)
{
    *from = (int)((long)n * id / nth);
    *to   = (int)((long)n * (id + 1) / nth);
}

static void *tp_worker(void *vp)
{
    tp_arg        *a = vp;
    tp_pool       *p = a->pool;
    unsigned long  seen = 0;

    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (!p->stop && p->gen == seen)
            pthread_cond_wait(&p->cv_work, &p->mu);
        if (p->stop) {
            pthread_mutex_unlock(&p->mu);
            return NULL;
        }
        seen = p->gen;
        tp_fn fn   = p->fn;
        void *arg  = p->arg;
        int   n    = p->n;
        int   nth  = p->nth;
        int   id   = a->id;
        pthread_mutex_unlock(&p->mu);

        int from, to;
        tp_slice(id, nth, n, &from, &to);
        if (from < to) fn(arg, from, to);

        pthread_mutex_lock(&p->mu);
        if (--p->busy == 0) pthread_cond_signal(&p->cv_done);
        pthread_mutex_unlock(&p->mu);
    }
}

/* Поднять пул на p->want потоков. Возвращает 0, если подняться не вышло —
 * тогда считаем в один поток, а не отказываем. */
static int tp_start(void)
{
    if (P.ready) return 0;
    /* tp_set никто не звал — берём то, что машина готова дать: все ядра,
     * если только DC_THREADS не велел иначе. Пул ленивый именно поэтому:
     * лёгкий узел, который только считает выражения, до него не дойдёт.
     * Поле в BSS стартует нулём, а ноль — это тоже «не задано». */
    if (P.want <= 0) P.want = tp_default();

    pthread_mutex_init(&P.mu, NULL);
    pthread_cond_init(&P.cv_work, NULL);
    pthread_cond_init(&P.cv_done, NULL);

    int nth = P.want;
    if (nth < 1) nth = 1;
    if (nth > TP_MAX_THREADS) nth = TP_MAX_THREADS;
    P.nth = nth;

    if (nth <= 1) {
        P.helpers = 0;
        P.ready   = 1;
        return 0;
    }

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, TP_STACK_BYTES);

    int started = 0;
    for (int i = 0; i < nth - 1; i++) {
        ARGS[i].pool = &P;
        ARGS[i].id   = i;
        if (pthread_create(&P.th[i], &attr, tp_worker, &ARGS[i]) != 0) break;
        started++;
    }
    pthread_attr_destroy(&attr);

    /* Часть помощников не поднялась — работаем тем, что есть. Лучше
     * восемь потоков, чем шестнадцать с половиной. */
    P.helpers = started;
    P.nth     = started + 1;
    P.ready   = 1;
    return started == nth - 1 ? 0 : -1;
}

static void tp_stop(void)
{
    if (!P.ready) return;
    pthread_mutex_lock(&P.mu);
    P.stop = 1;
    pthread_cond_broadcast(&P.cv_work);
    pthread_mutex_unlock(&P.mu);
    for (int i = 0; i < P.helpers; i++) pthread_join(P.th[i], NULL);
    pthread_cond_destroy(&P.cv_work);
    pthread_cond_destroy(&P.cv_done);
    pthread_mutex_destroy(&P.mu);
    int want = P.want;
    memset(&P, 0, sizeof P);
    P.want = want;            /* заданное пользователем переживает пересоздание */
}

int tp_default(void)
{
    const char *env = getenv("DC_THREADS");
    if (env && *env) {
        int n = atoi(env);
        if (n > 0) return n > TP_MAX_THREADS ? TP_MAX_THREADS : n;
        if (n < 0) return 1;
    }
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    return (int)(n > TP_MAX_THREADS ? TP_MAX_THREADS : n);
}

void tp_set(int nthreads)
{
    int want = nthreads == 0 ? tp_default() : nthreads;
    if (want < 1) want = 1;
    if (want > TP_MAX_THREADS) want = TP_MAX_THREADS;
    if (P.ready && P.nth == want) return;   /* пул уже того же размера */
    tp_stop();
    P.want = want;
}

int tp_threads(void)
{
    if (P.ready) return P.nth;
    /* пул ещё не поднят, но сколько он будет считать — известно: столько
     * велел tp_set, а если не велел, столько ядер у машины */
    return P.want > 0 ? P.want : tp_default();
}

int tp_worthwhile(int ops)
{
    return ops >= TP_MIN_OPS;
}

int tp_run(tp_fn fn, void *arg, int n)
{
    if (n <= 0) return 0;

    tp_start();

    /* помощников нет — делить не на что */
    if (P.helpers == 0) {
        fn(arg, 0, n);
        return 0;
    }

    pthread_mutex_lock(&P.mu);
    P.fn     = fn;
    P.arg    = arg;
    P.n      = n;
    P.nth    = P.helpers + 1;
    P.busy   = P.helpers;
    P.gen++;
    pthread_cond_broadcast(&P.cv_work);
    int nth = P.nth;
    pthread_mutex_unlock(&P.mu);

    /* Вызывающий не простаивает: он берёт свой кусок наравне с помощниками.
     * Иначе на 16 потоках 15 будут ждать, а считать будет один. */
    int from, to;
    tp_slice(nth - 1, nth, n, &from, &to);
    if (from < to) fn(arg, from, to);

    pthread_mutex_lock(&P.mu);
    while (P.busy > 0) pthread_cond_wait(&P.cv_done, &P.mu);
    pthread_mutex_unlock(&P.mu);
    return 1;
}

void tp_shutdown(void)
{
    tp_stop();
}

#endif /* DC_LITE */
