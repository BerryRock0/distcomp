/* Проверка пути, который чат делает при загрузке модели:
 * llm_model_tokens читает заголовок и кодирует промпт.
 *
 *   llm_tokens <model.gguf> [текст]
 */
#include "gguf.h"
#include "shard.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "model.gguf";
    const char *text = argc > 2 ? argv[2]
        : "<|im_start|>system\nты помощник<|im_end|>\n<|im_start|>user\nпривет<|im_end|>\n<|im_start|>assistant\n";
    int n = 0;
    char err[512] = {0};
    if (llm_model_tokens(path, text, &n, err, sizeof err) != 0) {
        fprintf(stderr, "ошибка: %s\n", err);
        return 2;
    }
    printf("токенов: %d\n", n);
    return 0;
}
