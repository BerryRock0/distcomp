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
void gguf_close(gguf_file *g);

/* Число по ключу и по ключу с префиксом архитектуры: gguf_u32(g, "block_count", "llama"). */
int  gguf_u32(const gguf_file *g, unsigned long *out, const char *key, const char *arch);
int  gguf_f32(const gguf_file *g, float *out, const char *key, const char *arch);
const char *gguf_str(const gguf_file *g, const char *key);
const gguf_meta *gguf_find(const gguf_file *g, const char *key);

const gguf_tensor *gguf_tensor_find(const gguf_file *g, const char *name);

/* Значение тензора как float: F32 копируется как есть, F16 разворачивается.
 * out должен вмещать count элементов. */
int gguf_tensor_to_float(const gguf_tensor *t, float *out, size_t count);

/* Разумный предел: файл больше — это не модель, а мусор. */
#define GGUF_MAX_SIZE   (8UL * 1024 * 1024 * 1024)
#define GGUF_MAX_TENSORS 4096
#define GGUF_MAX_META   4096

#endif /* DC_GGUF_H */