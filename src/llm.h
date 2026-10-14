#ifndef DC_LLM_H
#define DC_LLM_H

#include "conf.h"

#include <stddef.h>

/* Возвращает JSON плана (владение переходит вызывающему) либо NULL с заполненным err.
 * use_llm = 0 — сразу эвристика, не обращаясь к сети.
 * allow_fallback = 0 — при недоступной/сбойной LLM вернуть ошибку, а не эвристику.
 *
 * План не только запрашивается, но и проверяется: если модель вернула
 * мусор, цикл или несуществующую роль, причина возвращается ей же, и
 * модель пробует исправить план (DC_LLM_TRIES попыток). */
char *llm_make_plan(const dc_config *cfg, const char *goal, int use_llm, int allow_fallback,
                    char *err, size_t errlen);

/* Сколько раз пробуем спросить LLM, прежде чем сдаться. */
#define DC_LLM_TRIES 2

/* План, построенный без обращения к LLM. */
char *llm_heuristic_plan(const dc_config *cfg, const char *goal);

const char *llm_system_prompt(const dc_config *cfg);

/* Нужен ли ключ при обращении к этой LLM. Возвращает 0, если ключ есть
 * или сервер локальный/свой и задан auth = none. */
int  llm_needs_key(const dc_config *cfg);
int  llm_has_key(const dc_config *cfg);

/* Проверка LLM без построения плана: GET <url>/../models, разбор списка
 * моделей и внятная причина, если сервер недоступен.
 * llm_check печатает отчёт в stdout (CLI), llm_check_quiet молчит (панель). */
int  llm_check(const dc_config *cfg, char *err, size_t errlen);
int  llm_check_quiet(const dc_config *cfg, char *err, size_t errlen);

/* Список идентификаторов моделей сервера. Вызывающий освобождает и элементы,
 * и сам массив (free()). */
int  llm_list_models(const dc_config *cfg, char ***out, char *err, size_t errlen);

#endif /* DC_LLM_H */
