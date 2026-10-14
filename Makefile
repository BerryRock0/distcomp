CC      ?= gcc
CFLAGS  ?= -O2 -g
CFLAGS  += -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wno-unused-parameter -pthread
LDFLAGS +=
LDLIBS  += -lssl -lcrypto -lpthread -lm

SRC  := src
BUILD := build
BIN  := bin

COMMON_SRC := $(SRC)/util.c $(SRC)/json.c $(SRC)/http.c $(SRC)/conf.c \
              $(SRC)/plan.c $(SRC)/llm.c $(SRC)/sched.c $(SRC)/expr.c $(SRC)/udp.c \
              $(SRC)/tty.c $(SRC)/gguf.c $(SRC)/engine.c $(SRC)/tok.c
COMMON_OBJ := $(COMMON_SRC:$(SRC)/%.c=$(BUILD)/%.o)

CTL_OBJ   := $(BUILD)/main.o $(BUILD)/web.o $(BUILD)/ui.o $(BUILD)/shard.o $(BUILD)/tty.o
DAEMON_OBJ := $(BUILD)/daemon.o

ALL_OBJ := $(COMMON_OBJ) $(CTL_OBJ) $(DAEMON_OBJ)

TARGETS := $(BIN)/dcctl $(BIN)/dcd
TESTBIN := $(BIN)/expr_test
LITE    := $(BIN)/dcd-lite

# lite: то же самое, что dcd, но с маленькими буферами и -Os.
# Предназначен для SAMD51 (32K ОЗУ), RP2040, ESP32 — там, где большие
# буферы не помещаются. Код у обоих один и тот же: без потоков и malloc.
# DC_MAX_BODY — сколько байт задачи узел примет: 1 КБ ровно столько,
# сколько типовой приёмник MCU осилит держать в ОЗУ, дейтаграмма с запасом.
# DC_SHARD_MAX_BYTES — сколько весов модели узел готов держать на плате:
# 8 КБ против мегабайта пула у обычной сборки.
# -ffunction-sections/-fdata-sections работают только вместе с --gc-sections
# при линковке: без него компилятор готовит секции, а линкер их не выбрасывает.
LITEFLAGS := -Os -ffunction-sections -fdata-sections \
             -DDC_UDP_MAX_DGRAM=1200 -DDC_MAX_BODY=1024 -U_FORTIFY_SOURCE \
             -fno-stack-protector -DDC_LITE=1 \
             -DDC_SHARD_MAX_BYTES=8192 -DDC_SHARD_SCRATCH=256

# Custom includes (for extra flags/rules)
-include *.mk
.PHONY: all clean run-example config test lite size

all: $(TARGETS) $(TESTBIN) lite

$(BUILD) $(BIN):
	@mkdir -p $@

$(BUILD)/%.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN)/dcctl: $(COMMON_OBJ) $(CTL_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BIN)/dcd: $(COMMON_OBJ) $(DAEMON_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BIN)/expr_test: $(BUILD)/expr.o $(BUILD)/util.o tests/expr_test.c | $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/expr_test.c $(BUILD)/expr.o $(BUILD)/util.o $(LDLIBS)

lite: $(LITE)

$(LITE): $(BUILD)/daemon_lite.o $(BUILD)/util_lite.o $(BUILD)/json_lite.o \
         $(BUILD)/conf_lite.o $(BUILD)/expr_lite.o $(BUILD)/udp_lite.o \
         $(BUILD)/gguf_lite.o $(BUILD)/engine_lite.o $(BUILD)/tok_lite.o | $(BIN)
	$(CC) $(LITEFLAGS) -Wl,--gc-sections $(LDFLAGS) -o $@ $^ -lm

$(BUILD)/%_lite.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(LITEFLAGS) -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wno-unused-parameter \
	      -c $< -o $@

# размер lite-узла: главный ориентир при выборе устройства
size: $(LITE)
	@echo "== обычный узел (bin/dcd) =="; size bin/dcd | tail -1
	@echo "== lite-узел  (bin/dcd-lite) =="; size $(LITE) | tail -1
	@echo
	@$(MAKE) --no-print-directory lite-size-bytes

lite-size-bytes: $(LITE)
	@echo "текст lite-узла: $$(stat -c%s $(LITE)) байт (включая libc)"

# автозависимости по заголовкам
$(BUILD)/util.o:    $(SRC)/util.h
$(BUILD)/json.o:    $(SRC)/json.h  $(SRC)/util.h
$(BUILD)/http.o:    $(SRC)/http.h  $(SRC)/util.h
$(BUILD)/conf.o:    $(SRC)/conf.h  $(SRC)/util.h
$(BUILD)/plan.o:    $(SRC)/plan.h  $(SRC)/tty.h  $(SRC)/conf.h $(SRC)/json.h $(SRC)/util.h
$(BUILD)/llm.o:     $(SRC)/llm.h   $(SRC)/conf.h $(SRC)/plan.h $(SRC)/http.h $(SRC)/json.h $(SRC)/expr.h
$(BUILD)/sched.o:   $(SRC)/sched.h $(SRC)/tty.h $(SRC)/plan.h $(SRC)/conf.h $(SRC)/http.h $(SRC)/json.h
$(BUILD)/expr.o:    $(SRC)/expr.h  $(SRC)/util.h
$(BUILD)/udp.o:     $(SRC)/udp.h   $(SRC)/util.h
$(BUILD)/main.o:    $(SRC)/tty.h $(SRC)/conf.h $(SRC)/plan.h $(SRC)/llm.h $(SRC)/sched.h $(SRC)/util.h $(SRC)/web.h $(SRC)/shard.h
$(BUILD)/web.o:     $(SRC)/web.h  $(SRC)/conf.h $(SRC)/plan.h $(SRC)/llm.h $(SRC)/sched.h $(SRC)/json.h $(SRC)/util.h
$(BUILD)/ui.o:      $(SRC)/web.h
$(BUILD)/shard.o:   $(SRC)/shard.h $(SRC)/conf.h $(SRC)/plan.h $(SRC)/sched.h $(SRC)/json.h \
                    $(SRC)/util.h $(SRC)/engine.h $(SRC)/gguf.h
$(BUILD)/daemon.o:  $(SRC)/conf.h $(SRC)/json.h $(SRC)/expr.h $(SRC)/udp.h $(SRC)/util.h \
                    $(SRC)/engine.h $(SRC)/gguf.h
$(BUILD)/daemon_lite.o $(BUILD)/engine_lite.o $(BUILD)/gguf_lite.o $(BUILD)/tok_lite.o: \
                      $(SRC)/engine.h $(SRC)/gguf.h $(SRC)/tok.h

config: | $(BIN)
	@./$(BIN)/dcctl --help >/dev/null 2>&1 || true

# --- запуск локального стенда: 4 демона на одной машине ---------------------
run-example: all
	@./scripts/demo.sh

test: all
	@./scripts/test.sh

clean:
	rm -rf $(BUILD) $(BIN)
