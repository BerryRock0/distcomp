#ifndef DC_GGUF_H
#define DC_GGUF_H

#include <stddef.h>

/*
 * Чтение GGUF — формата, в котором llama.cpp, Ollama и vLLM хранят веса.
 *
 * Файл разбирается целиком в память вызывающего: парсер ничего не
 * распределяет сам. Тензоры не преобразуются — ссылки указывают прямо в
 * буфер файла вместе с типом хранения, поэтому устройство держит веса ровно
 * в том виде, в каком они лежат в модели (F32 или F16), и не тратит память
 * на «распухание» до четырёх байт на вес.
 */

typedef struct {
    const char     *name;
    unsigned        n_dims;
    unsigned long   dims[4];
    unsigned        type;      /* тип ggml: 0 = F32, 1 = F16 */
    size_t         off;       /* смещение от начала блока данных */
    const unsigned char *data; /* указатель внутрь буфера модели */
    size_t         nbytes;
} gguf_tensor;

/* Найденный ключ метаданных. Значение для чисел — в `num`, для строк — в `str`. */
typedef struct {
    char         *key;
    unsigned      type;       /* тип gguf_metadata_value_type */
    double        num;
    char         *str;
    /* массив значений: используется для tokenizer.ggml.tokens и .merges */
    unsigned      arr_type;
    size_t        arr_len;
    char        **arr_str;
    double       *arr_num;
} gguf_meta;

typedef struct {
    const unsigned char *file;
    size_t         size;
    unsigned       version;
    gguf_tensor   *tensors;
    size_t         n_tensors;
    gguf_meta     *meta;
    size_t         n_meta;
    unsigned       alignment;
} gguf_file;

/* Открывает файл и читает заголовок с описанием тензоров и метаданных.
 * Метаданные и строки копируются — вызывающий обязан вызвать gguf_free. */
int  gguf_open(gguf_file *g, const void *file, size_t size, char *err, size_t errlen);

/* То же, но size может быть меньше настоящего размера файла: читается только
 * начало (заголовок со словарём), а тензоры, чьи данные в переданный кусок не
 * попали, остаются без указателя на данные.
 *
 * Нужно, чтобы посчитать токены промпта, не читая гигабайты весов: обычный
 * gguf_open на таком префиксе отверг бы файл целиком — он справедливо ждёт
 * все веса, а здесь их и не было. На Qwen2.5 из-за этого словарь не читался
 * вовсе, и планирование молча уходило на эвристику. */
int  gguf_open_head(gguf_file *g, const void *file, size_t size, char *err, size_t errlen);

void gguf_close(gguf_file *g);

/* Число по ключу и по ключу с префиксом архитектуры: gguf_u32(g, "block_count", "llama"). */
int  gguf_u32(const gguf_file *g, unsigned long *out, const char *key, const char *arch);
int  gguf_f32(const gguf_file *g, float *out, const char *key, const char *arch);
const char *gguf_str(const gguf_file *g, const char *key);
const gguf_meta *gguf_find(const gguf_file *g, const char *key);

const gguf_tensor *gguf_tensor_find(const gguf_file *g, const char *name);

/* Тип первого квантованного тензора в модели (ggml: 10 = Q2_K, 11 = Q3_K,
 * 12 = Q4_K, 14 = Q6_K и т.д.) или -1, если все веса обычные F32/F16.
 *
 * Движок считает только F32 и F16: для квантованного веса он брал бы
 * «строку» данных с неверным шагом и молчал бы, выдав правдоподобный,
 * но бессмысленный результат. Поэтому квантованную модель отвергаем
 * заранее и внятно. */
int gguf_quant_type(const gguf_file *g);

/* Разворачивает элементы [off, off+n) тензора в float. Понимает F32, F16,
 * K-кванты (Q2_K..Q8_K) и старые кванты (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0).
 * Для квантов off обязан быть кратен размеру блока (256 или 32). */
int gguf_dequant(const gguf_tensor *t, float *dst, size_t off, size_t n);

/* Значение тензора как float: F32 копируется как есть, F16 разворачивается,
 * кванты деквантируются. out должен вмещать count элементов. */
int gguf_tensor_to_float(const gguf_tensor *t, float *out, size_t count);

/* Разумный предел: файл больше — это не модель, а мусор. */
#define GGUF_MAX_SIZE   (8UL * 1024 * 1024 * 1024)
#define GGUF_MAX_TENSORS 4096
#define GGUF_MAX_META   4096

#endif /* DC_GGUF_H */