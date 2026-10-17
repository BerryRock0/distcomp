#ifndef DC_LLM_H
#define DC_LLM_H

#include "conf.h"
#include "shard.h"
#include "util.h"

#include <stddef.h>

/*
 * Планирование. Модель — локальная и распределённая: она лежит не в этом
 * процессе, а по узлам роли shard, и планировщик лишь гоняет через них
 * активацию. Ни ключей, ни внешнего сервера здесь не бывает.
 */

/* Возвращает JSON плана (владение переходит вызывающему) либо NULL с
 * заполненным err.
 *   use_llm         = 0 — сразу эвристика, не открывая модель;
 *   allow_fallback  = 0 — при негодной/недоступной модели вернуть ошибку,
 *                     а не молча уйти на эвристику.
 *
 * План не только запрашивается, но и проверяется тем же кодом, что и
 * вручную написанный. Если модель вернула мусор, причина возвращается ей же,
 * и модель пробует исправить план (DC_LLM_TRIES попыток). */
char *llm_make_plan(const dc_config *cfg, const char *goal, int use_llm,
                    int allow_fallback, char *err, size_t errlen);

/* Сколько раз пробуем спросить модель, прежде чем сдаться. */
#define DC_LLM_TRIES 2

/* План, построенный без обращения к модели. */
char *llm_heuristic_plan(const dc_config *cfg, const char *goal);

const char *llm_system_prompt(const dc_config *cfg);

/* ----------------------------------------------------------------- чат --- */
/*
 * Веб-чат: та же модель и то же разложение по узлам, но системный промпт
 * другой (здесь модель отвечает человеку, а не строит план) и контекста
 * заказывается больше — разговор длиннее одной задачи.
 */

const char *llm_chat_system(const dc_config *cfg);

/* Промпт в формате ChatML (его понимают Qwen2 и Llama-3). history — уже
 * готовые реплики, каждая оформлена как «<|im_start|>роль\nтекст<|im_end|>». */
char *llm_chat_prompt(const char *system, const char *history, const char *message);

/* Дописать реплику в историю. */
void llm_chat_history_add(strbuf *hist, const char *role, const char *text);

/* Открыть конвейер под чат. Отличается от open_infer только объёмом
 * контекста: системный промпт, история разговора и длинный ответ. */
dc_infer *llm_open_chat(dc_config *cfg, int max_new, char *err, size_t errlen);

/* ---------------------------------------------------------- проверка --- */

/* Состояние локальной модели: что за файл, что за архитектура и влезет ли
 * он в парк узлов. Один раз собирается, дальше читают и CLI, и панель. */
typedef struct {
    int    ready;          /* 1 — модель разложилась, планы строить можно */
    char   model[512];     /* путь к файлу (или что задано в конфигурации) */
    char   arch[32];
    int    n_layer;
    int    n_embd;
    int    n_vocab;
    int    n_ctx;
    int    n_nodes;        /* узлов роли shard */
    int    n_stages;
    size_t bytes;
    char   note[256];      /* что мешает, если ready == 0 */
} llm_status;

int llm_local_status(const dc_config *cfg, llm_status *st);

/* Проверка без построения плана. llm_check печатает отчёт (CLI),
 * llm_check_quiet молчит (панель). */
int llm_check(const dc_config *cfg, char *err, size_t errlen);
int llm_check_quiet(const dc_config *cfg, char *err, size_t errlen);

#endif /* DC_LLM_H */