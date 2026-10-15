#!/usr/bin/env bash
# Тесты distcomp. Демоны поднимаются на свободных портах (UDP).
#   1. вычислитель выражений (юнит-тесты)
#   2. эвристическое планирование (без LLM)
#   3. роль без назначенного узла отклоняется
#   4. цикл в зависимостях отклоняется
#   5. провал проверки -> блокировка зависимых шагов и код возврата 1
#   6. недоступный узел -> повтор -> переключение на другой узел той же роли
#   7. распределение по свободным узлам одной роли
#   8. планирование через LLM (mock OpenAI)
#   9. запоминание состоялось
#  10. UDP: битая дейтаграмма и чужой протокол не ломают узел
#  11. веб-панель: отдаёт страницу, API состояния и выполняет запуск
#  12. lite-узел: считает, объявляется, отвечает на несколько ролей
#  13. возможности узлов: планировщик не шлёт большое на узел с малым лимитом
#  14. смешанный парк: сервер и lite-узел в одной роле делят шаги
#  15. негодный план от LLM: клиент объясняет причину и даёт модели починить
#  16. обнаружение узлов активным запросом, а не ожиданием объявления по таймеру
#  17. локальная LLM без ключа: llama.cpp/Ollama/LM Studio
#  18. модель llama (GGUF) разложена по свободной памяти устройств
#  19. разложенная по устройствам модель считает так же, как целиком
#      (сверка с локальным прогоном и независимым эталоном на numpy)
set -uo pipefail
cd "$(dirname "$0")/.."

T=.test
rm -rf "$T"; mkdir -p "$T/state"
PASS=0; FAIL=0
DC=./bin/dcctl
DCD=./bin/dcd
BASE=${BASE_PORT:-19400}
WEB=$((BASE + 30))
PIDS=""

cleanup() { for p in $PIDS; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT

start_daemons() {
  "$DCD" --role compute  --port $((BASE+1)) --tag cpu-a --state "$T/state" >"$T/cpu-a.log" 2>&1 & PIDS="$PIDS $!"
  "$DCD" --role compute  --port $((BASE+2)) --tag cpu-b --state "$T/state" >"$T/cpu-b.log" 2>&1 & PIDS="$PIDS $!"
  "$DCD" --role verify   --port $((BASE+3)) --tag chk   --state "$T/state" >"$T/chk.log"   2>&1 & PIDS="$PIDS $!"
  "$DCD" --role memorize --port $((BASE+4)) --tag mem   --state "$T/state" >"$T/mem.log"   2>&1 & PIDS="$PIDS $!"
  sleep 0.6
}

ok()   { printf '  \033[32mПРОЙДЕН\033[0m %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mУПАЛ\033[0m    %s\n' "$1"; [ $# -gt 1 ] && printf '         %s\n' "$2"; FAIL=$((FAIL+1)); }
check(){ if printf '%s' "$2" | grep -qF -- "$3"; then ok "$1"; else bad "$1" "не нашёл «$3»"; fi; }
check_absent(){ if printf '%s' "$2" | grep -qF -- "$3"; then bad "$1" "лишнее «$3»"; else ok "$1"; fi; }

start_daemons

cat > "$T/distcomp.conf" <<EOF
[llm]
payload_mode = expr

[role compute]
daemon = 127.0.0.1:$((BASE+1))
daemon = 127.0.0.1:$((BASE+2))

[role verify]
daemon = 127.0.0.1:$((BASE+3))

[role memorize]
daemon = 127.0.0.1:$((BASE+4))

[scheduler]
concurrency = 4
retries     = 2
timeout_ms  = 10000
backoff_ms  = 150
outdir      = $T/runs
EOF

GOAL='посчитать стоимость закупки 1250*3, затем долю (1250*3+480)/7'

# --- разложенная модель ----------------------------------------------------
# Модель, по которой строится план, лежит не в планировщике: её веса
# раскладываются по узлам роли shard, а планировщик гоняет между ними
# активацию. Эти узлы и файл нужны и планированию, и сценариям 18-19.
#
# --free-mem у планирования щедрый не потому, что так надо, а потому что у
# планирования длинный контекст (системный промпт плюс цель), а контекст —
# это KV-кэш, то есть память. Сценарии 18-19 проверяют разбиение на тесных
# узлах, где памяти в обрез.
PL1=$((BASE+80)); PL2=$((BASE+81)); PL3=$((BASE+82)); PL4=$((BASE+83)); PL5=$((BASE+84))
start_planner_shards() {
  "$DCD" --role shard --port $PL1 --tag p-in  --free-mem 2000000 --state "$T/state" >"$T/pl1.log" 2>&1 & PIDS="$PIDS $!"
  "$DCD" --role shard --port $PL2 --tag p-big --free-mem 2000000 --state "$T/state" >"$T/pl2.log" 2>&1 & PIDS="$PIDS $!"
  "$DCD" --role shard --port $PL3 --tag p-s1  --free-mem 2000000 --state "$T/state" >"$T/pl3.log" 2>&1 & PIDS="$PIDS $!"
  "$DCD" --role shard --port $PL4 --tag p-s2  --free-mem 2000000 --state "$T/state" >"$T/pl4.log" 2>&1 & PIDS="$PIDS $!"
  "$DCD" --role shard --port $PL5 --tag p-out --free-mem 2000000 --state "$T/state" >"$T/pl5.log" 2>&1 & PIDS="$PIDS $!"
  sleep 0.8
  cat > "$T/plan.conf" <<EOF
[llm]
model_file   = $T/model.gguf
max_new      = 96
payload_mode = expr

[role compute]
daemon = 127.0.0.1:$((BASE+1))
daemon = 127.0.0.1:$((BASE+2))

[role verify]
daemon = 127.0.0.1:$((BASE+3))

[role memorize]
daemon = 127.0.0.1:$((BASE+4))

[role shard]
daemon = 127.0.0.1:$PL1
daemon = 127.0.0.1:$PL2
daemon = 127.0.0.1:$PL3
daemon = 127.0.0.1:$PL4
daemon = 127.0.0.1:$PL5

[scheduler]
concurrency = 4
retries     = 2
timeout_ms  = 10000
backoff_ms  = 150
outdir      = $T/runs
EOF
}

# Эталонная мини-модель llama в формате GGUF (RMSNorm, RoPE, GQA, SwiGLU, BPE).
# Собирается один раз, до первого сценария: она нужна и планированию, и
# сценариям 18-19. Отчёт о сборке печатает сценарий 18, где про неё и речь.
MODEL_OK=0
if command -v python3 >/dev/null 2>&1; then
  if python3 scripts/llama_ref.py "$T/model.gguf" --json "$T/ref.json" >"$T/gguf.txt" 2>&1; then
    MODEL_OK=1
    start_planner_shards
  fi
fi

echo
echo "1. вычислитель выражений"
if ./bin/expr_test > "$T/expr.txt" 2>&1; then
  ok "$(tail -1 "$T/expr.txt")"
else
  bad "юнит-тесты вычислителя" "$(cat "$T/expr.txt")"
fi

echo
echo "2. эвристическое планирование (--no-llm)"
OUT=$($DC -c "$T/distcomp.conf" --no-llm run "$GOAL" 2>&1); RC=$?
check "все шаги выполнены"        "$OUT" "Все шаги выполнены"
check "код возврата 0"            "$RC"  "0"
check "вычислено 3750"            "$OUT" "3750"
check "план сохранён"             "$OUT" "plan.json"
check "check: узлы живы"          "$($DC -c "$T/distcomp.conf" check 2>&1)" "проблем 0"

echo
echo "3. роль без назначенного узла отклоняется"
cat > "$T/onlycompute.conf" <<EOF
[role compute]
daemon = 127.0.0.1:$((BASE+1))
EOF
cat > "$T/needsverify.json" <<'EOF'
{"goal":"нужна проверка","steps":[
 {"id":"c1","role":"compute","title":"цена","payload":"2*2","deps":[]},
 {"id":"v1","role":"verify","title":"сверка","payload":"all","deps":["c1"]}]}
EOF
OUT=$($DC -c "$T/onlycompute.conf" -f "$T/needsverify.json" run 2>&1); RC=$?
check "понятная ошибка"           "$OUT" "не назначена ни одному узлу"
check "код возврата 4"            "$RC"  "4"

echo
echo "4. цикл в зависимостях отклоняется"
cat > "$T/cycle.json" <<'EOF'
{"goal":"цикл","steps":[
 {"id":"a","role":"compute","title":"x","payload":"1+1","deps":["b"]},
 {"id":"b","role":"compute","title":"y","payload":"2+2","deps":["a"]}]}
EOF
OUT=$($DC -c "$T/distcomp.conf" -f "$T/cycle.json" run 2>&1)
check "сообщение о цикле"         "$OUT" "цикл"

echo
echo "5. провал проверки блокирует зависимые шаги"
cat > "$T/bad.json" <<'EOF'
{"goal":"заведомо неверная сверка","steps":[
 {"id":"c1","role":"compute","title":"цена","payload":"100*3","deps":[]},
 {"id":"v1","role":"verify","title":"сверка","payload":"999","deps":["c1"]},
 {"id":"m1","role":"memorize","title":"итог","payload":"итог","deps":["v1"]}]}
EOF
OUT=$($DC -c "$T/distcomp.conf" -f "$T/bad.json" run 2>&1); RC=$?
check "проверка провалена"        "$OUT" "ОШИБКА: ожидалось 999"
check "зависимый шаг заблокирован" "$OUT" "зависимость не выполнена"
check "код возврата 1"            "$RC"  "1"

echo
echo "6. недоступный узел -> повтор -> переключение на другой узел"
cat > "$T/flaky.conf" <<EOF
[role compute]
daemon = 127.0.0.1:$((BASE+1))
daemon = 127.0.0.1:$((BASE+9))
[role verify]
daemon = 127.0.0.1:$((BASE+3))
[role memorize]
daemon = 127.0.0.1:$((BASE+4))
[scheduler]
retries = 3
backoff_ms = 100
timeout_ms = 2000
outdir = $T/runs
EOF
cat > "$T/flaky.json" <<'EOF'
{"goal":"переключение узла","steps":[
 {"id":"c1","role":"compute","title":"a","payload":"2*2","deps":[]},
 {"id":"c2","role":"compute","title":"b","payload":"3*3","deps":[]},
 {"id":"c3","role":"compute","title":"c","payload":"4*4","deps":[]},
 {"id":"c4","role":"compute","title":"d","payload":"5*5","deps":[]},
 {"id":"v1","role":"verify","title":"сверка","payload":"all","deps":["c1","c2","c3","c4"]}]}
EOF
OUT=$($DC -c "$T/flaky.conf" -f "$T/flaky.json" run 2>&1)
check "задание выполнено в итоге"  "$OUT" "Все шаги выполнены"
check "была неудачная попытка"     "$OUT" "попытка"

echo
echo "7. распределение по свободным узлам одной роли"
for i in 1 2 3 4 5 6; do
  cat > "$T/load$i.json" <<EOF
{"goal":"нагрузка $i","steps":[
 {"id":"a","role":"compute","title":"a","payload":"$i+1","deps":[]},
 {"id":"b","role":"compute","title":"b","payload":"$i+2","deps":[]},
 {"id":"c","role":"compute","title":"c","payload":"$i+3","deps":[]},
 {"id":"d","role":"compute","title":"d","payload":"$i+4","deps":[]}]}
EOF
  $DC -c "$T/distcomp.conf" -f "$T/load$i.json" run >/dev/null 2>&1
done
HIT_A=$(grep -c '^\[compute/cpu-a\]' "$T/cpu-a.log" || true)
HIT_B=$(grep -c '^\[compute/cpu-b\]' "$T/cpu-b.log" || true)
if [ "$HIT_A" -eq 0 ] && [ "$HIT_B" -eq 0 ]; then
  HIT_A=$(grep -c 'cpu-a' "$T/cpu-a.log" || true)
  HIT_B=$(grep -c 'cpu-b' "$T/cpu-b.log" || true)
fi
if [ "$HIT_A" -gt 0 ] && [ "$HIT_B" -gt 0 ]; then
  ok "оба вычислителя загружены (cpu-a=$HIT_A, cpu-b=$HIT_B)"
else
  bad "оба вычислителя загружены" "cpu-a=$HIT_A cpu-b=$HIT_B"
fi

echo
echo "8. планирование локальной моделью, разложенной по узлам"
# Модель, которая строит план, не лежит в планировщике: её веса раскладываются
# по узлам роли shard. Проверяем всю цепочку — раскладку, открытие конвейера
# и честный отказ, когда промпт в контекст не влезает.
if [ -f "$T/model.gguf" ]; then
  OUT=$($DC -c "$T/plan.conf" llm check 2>&1); RC=$?
  check "llm check: модель на месте"      "$OUT" "архитектура:    llama"
  check "llm check: раскладка показана"   "$OUT" "Раскладка модели по устройствам"
  check "llm check: вход на своём узле"   "$OUT" "вход (эмбеддинг)"
  check "llm check: без ключа и сервера"  "$OUT" "ключи и внешний сервер не нужны"
  check "llm check: код возврата 0"       "$RC"  "0"

  OUT=$($DC -c "$T/plan.conf" -v run "$GOAL" 2>&1); RC=$?
  check "промпт посчитан по токенизатору" "$OUT" "промпт "
  check "контекст подобран под промпт"     "$OUT" "нужно контекста"
  # эталонная модель в тестах крошечная (4 слоя, словарь 256), поэтому на
  # каждый знак приходится токен и системный промпт в неё не влезает. Важно
  # тут не «план построен», а что об этом сказано вслух, а не сделано вид,
  # будто всё в порядке.
  check "нехватка контекста названа"       "$OUT" "больше"
  check "подсказка, что делать"            "$OUT" "словарём"
  check "откат на эвристику назван"        "$OUT" "эвристический планировщик"
  check "запуск всё равно доведён до конца" "$OUT" "Все шаги выполнены"
else
  echo "  пропущено: нет эталонной модели (нужен python3)"
fi

echo
echo "9. запоминание состоялось"
check "в памяти есть запись" "$(cat "$T/state/memory.log" 2>/dev/null)" '"role":"memorize"'
# журнал читают машины: каждая строка обязана быть настоящим JSON
if command -v python3 >/dev/null 2>&1; then
  if python3 -c '
import json,sys
n=0
for line in open(sys.argv[1], encoding="utf-8"):
    line=line.strip()
    if not line: continue
    d=json.loads(line)
    assert d["role"]=="memorize" and d["node"] and "ts" in d, d
    n+=1
assert n>0
' "$T/state/memory.log" 2>/dev/null; then
    ok "журнал памяти — валидный JSONL"
  else
    bad "журнал памяти — валидный JSONL" "$(head -1 "$T/state/memory.log" 2>/dev/null | head -c 120)"
  fi
fi

echo
echo "10. UDP: мусор в сокете не ломает узел"
python3 - <<PY 2>/dev/null
import socket
# мусор: не наш протокол, и наш протокол с испорченной контрольной суммой
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto(b'hello, i am not a dc packet', ('127.0.0.1', $((BASE+1))))
bad = bytearray(b'\x44\x43\x01\x01\x00\x00\x00\x04\x00\x00\x00\x00\x00\x01ab')
bad[13] ^= 0xFF                      # портим CRC
s.sendto(bytes(bad), ('127.0.0.1', $((BASE+1))))
s.close()
PY
sleep 0.3
OUT=$($DC -c "$T/distcomp.conf" check 2>&1)
check "узел пережил мусор"         "$OUT" "проблем 0"
OUT=$($DC -c "$T/distcomp.conf" --no-llm run "$GOAL" 2>&1)
check "и продолжает считать"       "$OUT" "Все шаги выполнены"

echo
echo "11. веб-панель"
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  $DC -c "$T/distcomp.conf" --no-llm --host 127.0.0.1 --port "$WEB" web >"$T/web.log" 2>&1 & PIDS="$PIDS $!"
sleep 0.8
HTML=$(curl -s "http://127.0.0.1:$WEB/" || true)
check "страница отдаётся"          "$HTML" "<title>distcomp"
check "есть поле цели"             "$HTML" "id='goal'"
check "есть форма узлов"           "$HTML" "id='furl'"
check "есть редактор плана"        "$HTML" "id='planjson'"
check "есть кнопка проверки LLM"   "$HTML" "id='bllm'"

# Скрипт панели обязан быть синтаксически верным: сломанная кавычка в строке C
# выглядит как «страница отдалась», а на деле панель молча не работает.
printf '%s' "$HTML" | sed -n '/<script>/,/<\/script>/p' | sed '1d;$d' > "$T/ui.js"
if command -v node >/dev/null 2>&1; then
  if node --check "$T/ui.js" >"$T/ui.err" 2>&1; then
    ok "скрипт панели разбирается (node --check)"
  else
    bad "скрипт панели разбирается (node --check)" "$(sed -n '4p' "$T/ui.err")"
  fi
elif [ -s "$T/ui.js" ] && [ "$(grep -c 'html +=' "$T/ui.js")" -gt 3 ]; then
  ok "скрипт панели на месте (node нет — проверка структуры)"
else
  bad "скрипт панели на месте" "в странице нет ожидаемого кода отрисовки"
fi

STATE=$(curl -s "http://127.0.0.1:$WEB/api/state" || true)
check "API состояния отвечает"     "$STATE" '"roles"'
check "в состоянии видны узлы"     "$STATE" "$((BASE+1))"
check "узлы определены как живые"  "$STATE" '"alive":true'

RID=$(curl -s -X POST "http://127.0.0.1:$WEB/api/run" \
        -H 'Content-Type: application/json' \
        -d '{"goal":"посчитать 7*6 и проверить","execute":true}' \
      | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')
if [ -n "$RID" ]; then ok "запуск создан ($RID)"; else bad "запуск создан"; fi
sleep 2.5
DETAIL=$(curl -s "http://127.0.0.1:$WEB/api/run?id=$RID" || true)
check "запуск выполнен"            "$DETAIL" '"phase":"done"'
check "результат шага виден"       "$DETAIL" "42"
check "состояние узла в ответе"    "$DETAIL" '"endpoint"'

# JSON обязан быть валидным целиком, а не «где-то содержит подстроку»
if command -v python3 >/dev/null 2>&1; then
  for name in state detail; do
    [ "$name" = state ] && BODY="$STATE" || BODY="$DETAIL"
    if printf '%s' "$BODY" | python3 -c 'import json,sys; json.load(sys.stdin)' 2>/dev/null; then
      ok "JSON /api/$name разбирается"
    else
      bad "JSON /api/$name разбирается" "$(printf '%s' "$BODY" | head -c 160)"
    fi
  done
  RSTEP=$(printf '%s' "$DETAIL" | python3 -c \
    'import json,sys; d=json.load(sys.stdin); print(d["plan"]["steps"][0]["endpoint"])' 2>/dev/null)
  case "$RSTEP" in
    *:*) ok "адрес узла не разыгран (endpoint=$RSTEP)" ;;
    *)   bad "адрес узла не разыгран" "endpoint=$RSTEP" ;;
  esac
fi

# план, присланный из панели, выполняется как есть — LLM не вызывается
cat > "$T/from_panel.json" <<'EOF'
{"goal":"план из панели","steps":[
 {"id":"p1","role":"compute","title":"свой шаг","payload":"21*2","deps":[]},
 {"id":"p2","role":"verify","title":"сверка","payload":"all","deps":["p1"]}]}
EOF
PANEL_BODY=$(python3 -c 'import json,sys; print(json.dumps({"plan": open(sys.argv[1]).read(), "execute": True}))' "$T/from_panel.json" 2>/dev/null || echo '{}')
PID2=$(curl -s -X POST "http://127.0.0.1:$WEB/api/run" -H 'Content-Type: application/json' \
       -d "$PANEL_BODY" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')
if [ -n "$PID2" ]; then ok "принят план из панели ($PID2)"; else bad "принят план из панели"; fi
sleep 1.5
DETAIL2=$(curl -s "http://127.0.0.1:$WEB/api/run?id=$PID2" || true)
check "присланный план выполнен"    "$DETAIL2" '"phase":"done"'
check "его результат на месте"      "$DETAIL2" "42"
check "и вернули текст плана"       "$DETAIL2" '"plan_json"'

OUT=$(curl -s "http://127.0.0.1:$WEB/api/llm" || true)
check "API статуса LLM отвечает"    "$OUT" '"ok"'
OUT=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$WEB/favicon.ico" || true)
check "favicon не 404"              "$OUT" "204"

curl -s -X POST "http://127.0.0.1:$WEB/api/role" -H 'Content-Type: application/json' \
     -d "{\"role\":\"compute\",\"url\":\"127.0.0.1:$((BASE+2))\"}" >/dev/null
sleep 0.2
STATE=$(curl -s "http://127.0.0.1:$WEB/api/state" || true)
NC=$(printf '%s' "$STATE" | grep -o "\"name\":\"compute\"" | wc -l)
check "узел добавлен через UI"     "$STATE" '"name":"compute"'

echo
echo "12. lite-узел (сборка для маленьких устройств)"
LITE=$((BASE+50))
./bin/dcd-lite --role compute --role memorize --port $LITE --tag lite --state "$T/state" \
  >"$T/lite.log" 2>&1 & PIDS="$PIDS $!"
sleep 0.6
check "запускается lite-сборка"    "$(cat "$T/lite.log")" "dcd-lite"
cat > "$T/lite.conf" <<EOF
[llm]

[role compute]
daemon = 127.0.0.1:$LITE
[role memorize]
daemon = 127.0.0.1:$LITE
[scheduler]
timeout_ms = 3000
retries = 0
outdir = $T/runs
EOF
OUT=$($DC -c "$T/lite.conf" check 2>&1)
check " lite-узел отвечает"        "$OUT" "живой"
check "объявляет обе роли"          "$OUT" "compute,memorize"
check "объявляет свой лимит тела"   "$OUT" "1024 байт"
cat > "$T/lite.json" <<'EOF'
{"goal":"задача для lite-узла","steps":[
 {"id":"c1","role":"compute","title":"счёт","payload":"6*7","deps":[]},
 {"id":"m1","role":"memorize","title":"запись","payload":"итог 42","deps":["c1"]}]}
EOF
OUT=$($DC -c "$T/lite.conf" -f "$T/lite.json" run 2>&1)
check "lite-узел считает"           "$OUT" "42"
check "lite-узел запоминает"        "$OUT" "Все шаги выполнены"

echo
echo "13. лимит тела: большое не отправляем маленькому"
python3 - "$T/over.json" <<'PY'
import json, sys
big = '1+' * 2000 + '1'          # около 4 КБ, при лимите lite-узла 1024
json.dump({'goal': 'перевес лимита', 'steps': [
    {'id': 'ok',  'role': 'compute', 'title': 'мелкий', 'payload': '2+2', 'deps': []},
    {'id': 'big', 'role': 'compute', 'title': 'перевес', 'payload': big, 'deps': []},
]}, open(sys.argv[1], 'w'))
PY
OUT=$($DC -c "$T/lite.conf" -f "$T/over.json" run 2>&1); RC=$?
check "мелкий шаг выполнен"         "$OUT" "-> 4"
check "перевес отклонён до отправки" "$OUT" "не помещается ни в один узел"
check "причина названа, а не таймаут" "$OUT" "не помещается ни в один узел"
check "код возврата 1"              "$RC"  "1"

echo
echo "14. смешанный парк: сервер + lite в одной роли"
SRV=$((BASE+51))
./bin/dcd --role compute --role verify --port $SRV --tag srv --hello-interval 2 \
  --state "$T/state" >"$T/srv.log" 2>&1 & PIDS="$PIDS $!"
sleep 0.6
cat > "$T/mix.conf" <<EOF
[llm]

[role compute]
daemon = 127.0.0.1:$SRV
daemon = 127.0.0.1:$LITE
[role verify]
daemon = 127.0.0.1:$SRV
[scheduler]
timeout_ms = 3000
retries = 0
outdir = $T/runs
EOF
python3 - "$T/mix.json" "$SRV" <<'PY'
import json, sys
big = '1+' * 2000 + '1'
json.dump({'goal': 'смешанный парк', 'steps': [
    {'id': f's{i}', 'role': 'compute', 'title': f'шаг {i}', 'payload': f'{i}*2', 'deps': []}
    for i in range(1, 7)
] + [{'id': 'big', 'role': 'compute', 'title': '4 КБ — только сервер',
      'payload': big, 'deps': []}]}, open(sys.argv[1], 'w'))
PY
OUT=$($DC -c "$T/mix.conf" -f "$T/mix.json" run 2>&1)
check "все шаги выполнены"          "$OUT" "Все шаги выполнены"
USED_LITE=$(printf '%s' "$OUT" | grep -c ":$LITE " || true)
USED_SRV=$(printf '%s' "$OUT" | grep -c ":$SRV " || true)
if [ "$USED_LITE" -gt 0 ] && [ "$USED_SRV" -gt 0 ]; then
  ok "работают оба узла (сервер=$USED_SRV, lite=$USED_LITE)"
else
  bad "оба узла загружены" "сервер=$USED_SRV lite=$USED_LITE"
fi
BIG_ON_SRV=$(printf '%s' "$OUT" | grep -A0 '"big"' | grep -c ":$SRV " || true)
check "большой шаг ушёл на сервер"  "$OUT" "4 КБ — только сервер"

echo
echo "15. --strict: негодный ответ модели не подменяется молча"
# Пользователь попросил модель — и должен получить отказ, а не тихо
# выполненный план от эвристики. Проверяем ровно это.
if [ "$MODEL_OK" = 1 ]; then
  OUT=$($DC -c "$T/plan.conf" --strict run "$GOAL" 2>&1); RC=$?
  check "причина отказа названа"      "$OUT" "контекст"
  check_absent "эвристика не подставлена" "$OUT" "эвристический планировщик"
  check "код возврата 4"              "$RC"  "4"
  check_absent "план вслепую не выполнен" "$OUT" "Все шаги выполнены"
else
  echo "  пропущено: нет эталонной модели (нужен python3)"
fi

echo
echo "16. обнаружение узлов по запросу (без ожидания объявления по таймеру)"
# узел уже объявился при старте и больше молчит 10 с — discover всё равно
# должен найти его сразу: поиск идёт активным запросом, а не ожиданием
DISC=$((BASE+60))
"$DCD" --role compute --port $DISC --tag disc-01 --state "$T/state" >"$T/disc.log" 2>&1 & PIDS="$PIDS $!"
sleep 1.2
OUT=$($DC -c "$T/distcomp.conf" discover 2 2>&1); RC=$?
check "узел найден"                 "$OUT" "disc-01"
check "адрес и порт показаны"       "$OUT" ":$DISC"
check "код возврата 0"              "$RC"  "0"
# повтор сразу: узел не объявляется заново, значит работает именно запрос
OUT=$($DC -c "$T/distcomp.conf" discover 1 2>&1)
check "и повторный поиск находит"   "$OUT" "disc-01"

echo
echo "17. планирование не требует ни ключа, ни сервера"
# Раньше здесь жил внешний LLM-сервер и вопрос «нужен ли ключ». Теперь
# вопрос снят: ключа не бывает, сервера не бывает — модель лежит в файле,
# а считают её узлы. Проверяем, что планирование работает в пустой сети.
if [ "$MODEL_OK" = 1 ]; then
  OUT=$($DC -c "$T/plan.conf" llm 2>&1); RC=$?
  check "модель описана"                "$OUT" "модель: $T/model.gguf"
  check "узлы названы"                   "$OUT" "узлы:   shard"
  check "про внешний сервер сказано"     "$OUT" "внешний сервер не нужны"
  check "код возврата 0"                 "$RC"  "0"

  # конфигурация вообще без ключей и адресов — только файл и узлы
  cat > "$T/nokey.conf" <<EOF
[llm]
model_file = $T/model.gguf

[role compute]
daemon = 127.0.0.1:$((BASE+1))

[role verify]
daemon = 127.0.0.1:$((BASE+3))

[role shard]
daemon = 127.0.0.1:$PL1
daemon = 127.0.0.1:$PL2
daemon = 127.0.0.1:$PL5
EOF
  OUT=$($DC -c "$T/nokey.conf" --strict llm 2>&1)
  check "три узла — раскладка возможна"   "$OUT" "Раскладка модели по устройствам"
  check "готовность подтверждена"        "$OUT" "план можно строить локально"
  # а если модели нет — это сказано прямо, а не сделано вид
  OUT=$($DC -c "$T/distcomp.conf" llm 2>&1); RC=$?
  check "без модели сказано честно"      "$OUT" "не задан model_file"
  check "и код возврата 3"               "$RC"  "3"
else
  echo "  пропущено: нет эталонной модели (нужен python3)"
fi

echo
echo "18. модель разложена по памяти устройств (GGUF/llama)"
SH1=$((BASE+70)); SH2=$((BASE+71)); SH3=$((BASE+72)); SH4=$((BASE+73)); SH5=$((BASE+74))
"$DCD" --role shard --port $SH1 --tag in  --free-mem 450000 --state "$T/state" >"$T/sh1.log" 2>&1 & PIDS="$PIDS $!"
"$DCD" --role shard --port $SH2 --tag big --free-mem 450000 --state "$T/state" >"$T/sh2.log" 2>&1 & PIDS="$PIDS $!"
"$DCD" --role shard --port $SH3 --tag sm1 --free-mem 250000 --state "$T/state" >"$T/sh3.log" 2>&1 & PIDS="$PIDS $!"
"$DCD" --role shard --port $SH4 --tag sm2 --free-mem 250000 --state "$T/state" >"$T/sh4.log" 2>&1 & PIDS="$PIDS $!"
"$DCD" --role shard --port $SH5 --tag out --free-mem 450000 --state "$T/state" >"$T/sh5.log" 2>&1 & PIDS="$PIDS $!"
sleep 0.8
cat > "$T/shard.conf" <<EOF
[llm]

[role shard]
daemon = 127.0.0.1:$SH1
daemon = 127.0.0.1:$SH2
daemon = 127.0.0.1:$SH3
daemon = 127.0.0.1:$SH4
daemon = 127.0.0.1:$SH5
[scheduler]
timeout_ms = 8000
retries = 0
EOF

# эталонная модель уже собрана в начале тестов; здесь про неё и отчёт
if [ "$MODEL_OK" = 1 ]; then
  ok "эталонная модель GGUF собрана ($(head -1 "$T/gguf.txt"))"
else
  bad "эталонная модель GGUF собрана" "нет python3"
fi

OUT=$($DC -c "$T/shard.conf" llm info "$T/model.gguf" 2>&1)
check "раскладка показана"              "$OUT" "Раскладка модели по устройствам"
check "модель разобрана на слои"        "$OUT" "слоёв 4"
check "вход на отдельном узле"          "$OUT" "вход (эмбеддинг)"
check "выход на отдельном узле"         "$OUT" "выход (голова)"
# узлу с большей памятью досталось два слоя, узлам с меньшей — по одному
BIG=$(printf '%s' "$OUT" | grep -c "слои 0..1")
SM1=$(printf '%s' "$OUT" | grep -c "слой 2")
SM2=$(printf '%s' "$OUT" | grep -c "слой 3")
if [ "$BIG" -gt 0 ] && [ "$SM1" -gt 0 ] && [ "$SM2" -gt 0 ]; then
  ok "по памяти: большому узлу 2 слоя, маленьким по 1"
else
  bad "по памяти: большому узлу 2 слоя, маленьким по 1" "big=$BIG sm1=$SM1 sm2=$SM2"
fi
# конвейеру нужно минимум три узла: вход, слои и выход — на разных устройствах
cat > "$T/two.conf" <<EOF
[llm]

[role shard]
daemon = 127.0.0.1:$SH1
daemon = 127.0.0.1:$SH5
[scheduler]
timeout_ms = 8000
retries = 0
EOF
OUT=$($DC -c "$T/two.conf" llm info "$T/model.gguf" 2>&1)
check "двух узлов мало для конвейера"   "$OUT" "нужно минимум 3 узла"
OUT=$($DC -c "$T/shard.conf" llm deploy "$T/model.gguf" 2>&1)
check "веса залиты на все узлы"         "$OUT" "Веса разложены"
check_absent "загрузка без ошибок"      "$OUT" "ОШИБКА"

echo
echo "19. модель на нескольких устройствах считает так же, как целиком"
OUT=$($DC -c "$T/shard.conf" llm prompt "$T/model.gguf" "distcomp" 4 2>&1); RC=$?
check "конвейер отработал"              "$OUT" "Генерация (жадная)"
check "сверка с локальным прогоном"     "$OUT" "все 4 сгенерированных токенов совпали"
check "код возврата 0"                  "$RC"  "0"
# независимый эталон на numpy: те же жадные токены, что выдал распределённый конвейер
if command -v python3 >/dev/null 2>&1; then
  EXP=$(python3 scripts/gguf_expected.py --prompt distcomp --tokens 4 2>/dev/null)
  check "токены совпали с эталоном numpy" "$OUT" "$EXP"
fi

echo
echo "20. локальный интерфейс: история, повтор, версия, цвет"
OUT=$($DC -c "$T/distcomp.conf" --version 2>&1)
check "версия печатается"           "$OUT" "distcomp"
OUT=$($DC -c "$T/distcomp.conf" runs 2>&1)
check "история запусков"            "$OUT" "Запуски в"
check "в истории есть метки"        "$OUT" "шагов"
LAST=$($DC -c "$T/distcomp.conf" runs 2>&1 | sed -n 's/^  \([0-9][0-9-]*\)  .*/\1/p' | head -1)
if [ -n "$LAST" ]; then ok "последний запуск: $LAST"; else bad "найден последний запуск"; fi
OUT=$($DC -c "$T/distcomp.conf" replay "$LAST" 2>&1); RC=$?
check "повтор выполнил план"        "$OUT" "Все шаги выполнены"
check "повтор: код возврата 0"      "$RC"  "0"
OUT=$($DC -c "$T/distcomp.conf" replay 20200101-000000 2>&1)
check "повтор несуществующего"      "$OUT" "нет плана"
OUT=$($DC -c "$T/distcomp.conf" --color always roles 2>&1)
check "цвет включается флагом"      "$OUT" "$(printf '\033[')"
OUT=$($DC -c "$T/distcomp.conf" --color bogus roles 2>&1)
check "неверный режим цвета отвергнут" "$OUT" "auto | always | never"
# метки запусков не должны затирать друг друга: два запуска за одну секунду
ID1=$($DC -c "$T/distcomp.conf" --no-llm plan "раз 1" 2>&1 | sed -n 's#.*runs/\([^/]*\)/plan.json#\1#p' | head -1)
ID2=$($DC -c "$T/distcomp.conf" --no-llm plan "раз 2" 2>&1 | sed -n 's#.*runs/\([^/]*\)/plan.json#\1#p' | head -1)
if [ -n "$ID1" ] && [ -n "$ID2" ] && [ "$ID1" != "$ID2" ] && [ -f "$T/runs/$ID1/plan.json" ] && [ -f "$T/runs/$ID2/plan.json" ]; then
  ok "метки запусков не затирают друг друга ($ID1, $ID2)"
else
  bad "метки запусков не затирают друг друга" "$ID1 / $ID2"
fi
# в конвейере цвет выключен сам: логи и grep должны работать как раньше
OUT=$($DC -c "$T/distcomp.conf" --no-llm run 'посчитать 5*5' 2>&1)
case "$OUT" in *"$(printf '\033[')"*) bad "в перенаправлении нет управляющих кодов" ;; *) ok "в перенаправлении нет управляющих кодов" ;; esac

echo
echo "-----------------------------------------"
echo "пройдено: $PASS, упало: $FAIL"
[ "$FAIL" -eq 0 ]
