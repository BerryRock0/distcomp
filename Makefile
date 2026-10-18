CC      ?= gcc
CFLAGS  ?= -O2 -g
CFLAGS  += -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wno-unused-parameter -pthread
LDFLAGS +=
# OpenSSL больше не нужен: HTTP-клиент остался только ради обращения к чужому
# LLM-серверу, а планирование теперь ведёт локальная модель, разложенная по
# узлам. Сборка стала без единой внешней библиотеки.
LDLIBS  += -lpthread -lm

SRC  := src
BUILD := build
BIN  := bin

COMMON_SRC := $(SRC)/util.c $(SRC)/json.c $(SRC)/conf.c \
              $(SRC)/plan.c $(SRC)/llm.c $(SRC)/sched.c $(SRC)/expr.c $(SRC)/udp.c \
              $(SRC)/tcp.c $(SRC)/dtn.c $(SRC)/xport.c $(SRC)/shard.c \
              $(SRC)/tty.c $(SRC)/gguf.c $(SRC)/engine.c $(SRC)/tok.c \
              $(SRC)/thread.c
COMMON_OBJ := $(COMMON_SRC:$(SRC)/%.c=$(BUILD)/%.o)

# shard.c попал в общие объекты потому, что планирование зовёт его движок:
# модель, разложенная по узлам, теперь строит план сама, а не просить его
# у внешнего сервера.
CTL_OBJ   := $(BUILD)/main.o $(BUILD)/web.o $(BUILD)/ui.o
DAEMON_OBJ := $(BUILD)/daemon.o

ALL_OBJ := $(COMMON_OBJ) $(CTL_OBJ) $(DAEMON_OBJ)

TARGETS := $(BIN)/dcctl $(BIN)/dcd
TESTBIN := $(BIN)/expr_test
# Деквантизация против эталона ggml. tests/dequant_test проверяет правдоподобность
# (нет NaN, разброс в разумных пределах) — этого недостаточно: переставленные
# веса остаются числами того же порядка, и неверная раскладка блока проходит
# незаметно. Здесь сравниваются все 256 весов блока с эталоном.
QUANTBIN := $(BIN)/quant_test
QUANTREAL := $(BIN)/quant_real
# Служебные токены <|im_start|>: через BPE такой текст распадается на
# обычные куски, и модель не видит границ реплик.
SPECIALBIN := $(BIN)/tok_special
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
.PHONY: all clean distclean run-example config test lite size

all: $(TARGETS) $(TESTBIN) $(QUANTBIN) $(QUANTREAL) $(SPECIALBIN) lite

$(SPECIALBIN): $(BUILD)/tok.o $(BUILD)/gguf.o $(BUILD)/util.o \
               tests/tok_special.c | $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) -I$(SRC) -o $@ tests/tok_special.c \
	      $(BUILD)/tok.o $(BUILD)/gguf.o $(BUILD)/util.o $(LDLIBS)

$(QUANTBIN): $(BUILD)/gguf.o $(BUILD)/util.o tests/quant_test.c | $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) -I$(SRC) -o $@ tests/quant_test.c \
	      $(BUILD)/gguf.o $(BUILD)/util.o $(LDLIBS)

# blocks.bin снимает scripts/dump_blocks.py: настоящие блоки из модели плюс
# значения ggml. Нужен python3 с пакетом gguf; без него цель просто не
# собирается, и test.sh этот сценарий пропускает.
$(QUANTREAL): $(BUILD)/gguf.o $(BUILD)/util.o tests/quant_real.c | $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) -I$(SRC) -o $@ tests/quant_real.c \
	      $(BUILD)/gguf.o $(BUILD)/util.o $(LDLIBS)

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

# Лёгкий узел. Только UDP: TCP и DTN — потоковые транспорты, а на
# микроконтроллере их нет и не нужно. И без модели: движок занимает 23 КБ
# текста плюс пул в ОЗУ, а запускать его там нечем. Поэтому в списке
# объектов нет ни gguf, ни engine, ни tok — не dead code, который линкер
# выбросит, а вовсе ненужные исходники.
#
# Лёгкий узел отдаёт free_mem = 0, поэтому распределитель сам не положит
# на него слои модели, а если задача придёт — откажет внятно.
LITE_OBJ := $(BUILD)/daemon_lite.o $(BUILD)/util_lite.o $(BUILD)/json_lite.o \
            $(BUILD)/conf_lite.o $(BUILD)/expr_lite.o $(BUILD)/udp_lite.o \
            $(BUILD)/thread_lite.o

$(LITE): $(LITE_OBJ) | $(BIN)
	$(CC) $(LITEFLAGS) -Wl,--gc-sections $(LDFLAGS) -o $@ $^ -lm

$(BUILD)/%_lite.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(LITEFLAGS) -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wno-unused-parameter \
	      -c $< -o $@

# ================================================== виды и архитектуры ===
#
# Узел бывает двух видов, и разница между ними не в размере буфера, а в том,
# что узел вообще умеет:
#
#   heavy  bin/dcd       сервер, Raspberry Pi, рабочая станция. Держит части
#                        модели, считает на всех свободных ядрах, говорит по
#                        UDP, TCP и DTN.
#
#   light  bin/dcd-lite  микроконтроллер: SAMD51, RP2040, ESP32, STM32.
#                        Выражения, проверка, память. Без модели, без
#                        потоков, только UDP. 50 КБ против 750 КБ.
#
# Под каждую архитектуру, для которой есть компилятор, — своя пара.
# Кладём в bin/<arch>/: при прошивке копируется та папка, которая устройству
# подходит, и бинарник чужой архитектуры на доску по ошибке не уедет.
#
#   make aarch64          собрать aarch64 (оба вида)
#   make aarch64-light    собрать один конкретный вариант
#   make cross            все архитектуры, для которых есть компилятор
#   make arches           показать, что есть и чего не хватает

ARCHS := x86_64 aarch64 arm riscv64 riscv32 avr

# Префикс кросс-компилятора. Пустой — сборка на своей машине: тогда DC_ARCH
# определится сам, по макросам компилятора.
TOOLCHAIN_x86_64  :=
TOOLCHAIN_aarch64 := aarch64-linux-gnu-
TOOLCHAIN_arm     := arm-linux-gnueabihf-
TOOLCHAIN_riscv64 := riscv64-unknown-elf-
TOOLCHAIN_riscv32 := riscv32-unknown-elf-
TOOLCHAIN_avr     := avr-

# Своя машина: её имя идёт первым, и именно её сборка называется bin/dcd.
HOST_ARCH := $(patsubst %-,%,$(notdir $(shell $(CC) -dumpmachine 2>/dev/null)))

# Есть ли компилятор для этой архитектуры. Проверка на уровне make, а не в
# shell: в цикле оболочки подставить аргумент в $(call) нельзя, а через make
# — можно, и весь список решается один раз, а не на каждой строке.
have-arch = $(shell command -v $(TOOLCHAIN_$(1))gcc >/dev/null 2>&1 && echo yes)
ARCHS_OK  := $(foreach a,$(ARCHS),$(if $(call have-arch,$(a)),$(a)))

# Какой цели звать, чтобы получить узел этого вида. Путь указываем целиком:
# в подсборке bin уже переопределён, и короткое имя там не найдётся.
FLAVOUR_TARGET_heavy = $(BIN)/$(1)/dcd
FLAVOUR_TARGET_light = $(BIN)/$(1)/dcd-lite

# Один вариант: рекурсивный make подменяет только BUILD, BIN и CC. Сами
# правила сборки остаются одни и те же, поэтому «как собирается узел» не
# может разойтись с обычной сборкой — и правила не приходится дублировать.
define cross_build
	@$(MAKE) --no-print-directory BUILD='$(BUILD)/$(1)' BIN='$(BIN)/$(1)' \
	    CC='$(TOOLCHAIN_$(1))gcc' $(call FLAVOUR_TARGET_$(2),$(1))
endef

define arch_flavour
$(1)-$(2): | $(BIN)/$(1)
	$$(call cross_build,$(1),$(2))
.PHONY: $(1)-$(2)
endef

$(foreach a,$(ARCHS),$(eval $(call arch_flavour,$(a),heavy)))
$(foreach a,$(ARCHS),$(eval $(call arch_flavour,$(a),light)))

# по одному правилу mkdir на архитектуру: объявлять его внутри define нельзя,
# цикл зовёт его дважды — по разу на вид, и make ругается на перекрытие рецепта
$(foreach a,$(ARCHS),$(eval $(BIN)/$(a): ; @mkdir -p $$@))

# make aarch64 — оба вида сразу. Без компилятора — внятная ошибка со списком
# того, что можно поставить, а не «command not found» из make.
.PHONY: $(ARCHS)
$(ARCHS):
	@if [ -z '$(call have-arch,$@)' ]; then \
	  echo "для $@ нужен $(TOOLCHAIN_$@)gcc — такого компилятора нет."; \
	  echo "поставить можно так (debian/ubuntu):"; \
	  echo "  aarch64: gcc-aarch64-linux-gnu"; \
	  echo "  arm:     gcc-arm-linux-gnueabihf"; \
	  echo "  riscv:   gcc-riscv64-unknown-elf"; \
	  echo "  avr:     gcc-avr"; \
	  echo "готовы: $(ARCHS_OK)"; \
	  exit 1; \
	fi
	@$(MAKE) --no-print-directory $@-heavy $@-light

.PHONY: cross arches
cross:
	@$(foreach a,$(ARCHS), \
	   if [ -n '$(call have-arch,$(a))' ]; then \
	     echo '== $(a) ($(TOOLCHAIN_$(a))gcc)'; \
	     $(MAKE) --no-print-directory $(a) || exit 1; \
	   else \
	     echo '== $(a): компилятора нет — пропускаю ($(TOOLCHAIN_$(a))gcc)'; \
	   fi; )
	@echo
	@echo 'собрано архитектур: $(words $(ARCHS_OK)) из $(words $(ARCHS))'

arches:
	@printf '%-11s %-24s %-8s %s\n' 'архитектура' 'компилятор' 'статус' 'файлы'
	@$(foreach a,$(ARCHS), \
	   printf '%-11s %-24s %-8s %s\n' '$(a)' '$(TOOLCHAIN_$(a))gcc' \
	     '$(if $(call have-arch,$(a)),есть,— нет)' \
	     '$(if $(wildcard $(BIN)/$(a)/dcd),dcd dcd-lite,—)'; )

# Размеры узлов. Главный ориентир при выборе устройства: тяжёлый узел на
# сервер, лёгкий — на плату с десятками килобайт ОЗУ.
size: $(TARGETS) $(LITE)
	@printf '\n== %s ==\n' 'тяжёлый узел (модель, потоки, TCP/DTN)'
	@size $(BIN)/dcd | tail -1
	@printf '   файла: %s байт\n\n' "$$(stat -c%s $(BIN)/dcd)"
	@printf '== %s ==\n' 'лёгкий узел (выражения, только UDP)'
	@size $(LITE) | tail -1
	@printf '   файла: %s байт\n\n' "$$(stat -c%s $(LITE))"
	@$(foreach a,$(ARCHS_OK), \
	   if [ -f '$(BIN)/$(a)/dcd' ]; then \
	     printf '== %s ==\n' '$(a)'; \
	     printf '   dcd      %s байт\n' "$$(stat -c%s '$(BIN)/$(a)/dcd')"; \
	     printf '   dcd-lite %s байт\n' "$$(stat -c%s '$(BIN)/$(a)/dcd-lite')"; \
	   fi; )
	@$(MAKE) --no-print-directory arches

lite-size-bytes: $(LITE)
	@echo "текст lite-узла: $$(stat -c%s $(LITE)) байт (включая libc)"

# автозависимости по заголовкам
$(QUANTBIN) $(QUANTREAL): $(SRC)/gguf.h $(SRC)/util.h
$(SPECIALBIN): $(SRC)/tok.h $(SRC)/gguf.h $(SRC)/util.h
$(BUILD)/util.o:    $(SRC)/util.h
$(BUILD)/json.o:    $(SRC)/json.h  $(SRC)/util.h
$(BUILD)/conf.o:    $(SRC)/conf.h  $(SRC)/util.h
$(BUILD)/plan.o:    $(SRC)/plan.h  $(SRC)/tty.h  $(SRC)/conf.h $(SRC)/json.h $(SRC)/util.h
$(BUILD)/expr.o:    $(SRC)/expr.h  $(SRC)/util.h
$(BUILD)/udp.o:     $(SRC)/udp.h   $(SRC)/util.h
$(BUILD)/tcp.o:     $(SRC)/tcp.h   $(SRC)/util.h
$(BUILD)/dtn.o:     $(SRC)/dtn.h   $(SRC)/tcp.h $(SRC)/util.h
$(BUILD)/xport.o:   $(SRC)/xport.h $(SRC)/conf.h $(SRC)/udp.h $(SRC)/tcp.h $(SRC)/dtn.h $(SRC)/util.h
$(BUILD)/llm.o:     $(SRC)/llm.h   $(SRC)/conf.h $(SRC)/plan.h $(SRC)/shard.h $(SRC)/json.h $(SRC)/util.h $(SRC)/expr.h
$(BUILD)/sched.o:   $(SRC)/sched.h $(SRC)/tty.h $(SRC)/plan.h $(SRC)/conf.h $(SRC)/xport.h $(SRC)/json.h
$(BUILD)/main.o:    $(SRC)/tty.h $(SRC)/conf.h $(SRC)/plan.h $(SRC)/llm.h $(SRC)/sched.h $(SRC)/util.h $(SRC)/web.h $(SRC)/shard.h
$(BUILD)/web.o:     $(SRC)/web.h  $(SRC)/conf.h $(SRC)/plan.h $(SRC)/llm.h $(SRC)/sched.h $(SRC)/json.h $(SRC)/util.h $(SRC)/shard.h
$(BUILD)/ui.o:      $(SRC)/web.h
$(BUILD)/shard.o:   $(SRC)/shard.h $(SRC)/conf.h $(SRC)/sched.h $(SRC)/json.h $(SRC)/xport.h \
                    $(SRC)/util.h $(SRC)/engine.h $(SRC)/gguf.h
$(BUILD)/daemon.o:  $(SRC)/conf.h $(SRC)/json.h $(SRC)/expr.h $(SRC)/udp.h $(SRC)/tcp.h $(SRC)/dtn.h $(SRC)/util.h \
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

# clean убирает только то, что сам же и собрал: объекты и бинарники.
# Модели и конфигурация рядом не трогаются — раньше здесь стояло
# «rm -rf build bin», и одна команда clean стирала скачанную модель
# на полгигабайта, которую заново приходилось качать с сети.
clean:
	rm -rf $(BUILD)
	rm -f $(BIN)/dcctl $(BIN)/dcd $(BIN)/dcd-lite
	rm -rf $(foreach a,$(ARCHS),$(BIN)/$(a))
	rm -f $(BIN)/expr_test $(BIN)/quant_test $(BIN)/quant_real $(BIN)/gen_test
	rm -f $(BIN)/dump_test $(BIN)/engine_test $(BIN)/tok_special $(BIN)/llm_tokens

distclean: clean
	rm -rf $(BUILD) $(BIN)
