#ifndef DC_SHARD_H
#define DC_SHARD_H

#include "conf.h"

#include <stddef.h>

/*
 * Модель, разложенная по устройствам.
 *
 * Вес любой модели — это почти целиком VRAM: на Raspberry Pi с 8 ГБ не
 * помещается то, что на двух серверах помещается, а микроконтроллер не
 * вмещает и одного слоя. Поэтому модель режется на слои, слои раскладываются
 * по узлам пропорционально их свободной памяти, и дальше по сети идёт не
 * модель, а активация: каждый узел применяет свои слои и отдаёт результат.
 *
 * Планировщик держит на себе только текст промпта и одно состояние из n_embd
 * чисел. Сама модель целиком в памяти планировщика не нужна — поэтому он
 * годится и как узел, у которого для модели памяти нет вовсе.
 */

/* ------------------------------------------------------------- инференс --- */

typedef struct dc_infer dc_infer;

typedef struct {
    int n_ctx;     /* сколько токенов контекста заготовить; 0 — взять из модели */
    int verbose;   /* печатать раскладку и ход генерации */
} dc_infer_opts;

/* Открывает модель GGUF, раскладывает стадии по узлам роли shard (память
 * балансируется по их объявленному free_mem) и заливает веса.
 * Возвращает 0 и готовый конвейер в *out; иначе причина в err. */
int dc_infer_open(dc_config *cfg, const char *model_path, const dc_infer_opts *o,
                  dc_infer **out, char *err, size_t errlen);

/* То же, но без заливки весов: проверка «поместится ли модель в парк узлов».
 * Сеть при этом не дёргается — спрашиваются только PING. */
int dc_infer_layout(dc_config *cfg, const char *model_path, const dc_infer_opts *o,
                    dc_infer **out, char *err, size_t errlen);

/* Жадно (argmax) генерирует текст после prompt, не больше max_new токенов.
 * Возвращает malloc-строку либо NULL с причиной в err.
 * Контекст сбрасывается перед каждым вызовом, поэтому один и тот же dc_infer
 * можно переиспользовать для нескольких промптов подряд. */
char *dc_infer_generate(dc_infer *in, const char *prompt, int max_new,
                        char *err, size_t errlen);

/* На каждый новый токен вызывается cb с его id и куском текста (только тем,
 * что ещё не отдавали). cb возвращает 0, чтобы остановить генерацию.
 * Нужен веб-чату: ответ показывают по мере поступления, а не после конца.
 * Полный ответ возвращается в любом случае. */
typedef int (*dc_tok_cb)(int tok, const char *text, void *ud);

char *dc_infer_generate_cb(dc_infer *in, const char *prompt, int max_new,
                           dc_tok_cb cb, void *ud, char *err, size_t errlen);

/* Сколько токенов контекста занял последний вызов generate. */
int dc_infer_used(const dc_infer *in);

void dc_infer_free(dc_infer *in);

/* Сведения о модели — для отчётов в CLI и панели. */
const char *dc_infer_model_path(const dc_infer *in);
const char *dc_infer_arch(const dc_infer *in);
int         dc_infer_n_layer(const dc_infer *in);
int         dc_infer_n_embd(const dc_infer *in);
int         dc_infer_n_vocab(const dc_infer *in);
int         dc_infer_n_ctx(const dc_infer *in);
int         dc_infer_n_nodes(const dc_infer *in);
int         dc_infer_n_stages(const dc_infer *in);
size_t      dc_infer_model_bytes(const dc_infer *in);
int         dc_infer_vocab_open(const dc_infer *in);

/* Печатает раскладку по узлам — то же, что делает `llm info`. */
void dc_infer_print_layout(const dc_infer *in);

/* Сколько токенов в тексте по токенизатору модели.
 *
 * Считается по-настоящему, а не «три знака на токен»: у настоящей модели со
 * словарём в сотни тысяч слов это примерно так и есть, а у модели со
 * словарём в 256 слов каждый знак — отдельный токен, и оценка врёт в семь
 * раз. Планировщик обязан знать размер промпта точно, иначе он заранее
 * закажет у узлов контекст не того размера.
 *
 * Читается только заголовок GGUF с токенизатором, а не весь файл: веса всё
 * равно лежат на узлах, и тянуть их в планировщик незачем. */
int llm_model_tokens(const char *model_path, const char *text, int *out,
                     char *err, size_t errlen);

/* ------------------------------------------------------------- команды --- */

/* Команда `dcctl llm ...`. Печатает отчёт, возвращает код возврата dcctl. */
int shard_cmd(dc_config *cfg, int argc, char **argv, char *err, size_t errlen);

#endif /* DC_SHARD_H */