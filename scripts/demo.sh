#!/usr/bin/env bash
# Стенд: обычный узел и lite-узел на одной машине (UDP), план без LLM.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=.demo
rm -rf "$RUN"; mkdir -p "$RUN/state"

# обычный узел: считает всё, две роли
./bin/dcd --role compute --role verify --port 9101 --tag server-a \
         --state "$RUN/state" >"$RUN/server-a.log" 2>&1 &
A=$!
# lite-узел: сборка для маленьких устройств, лимит тела 1024 байта.
# он же держит часть весов модели — на 8 КБ пула два-три слоя поместятся
./bin/dcd-lite --role compute --role memorize --role shard --port 9102 --tag mcu-01 \
         --free-mem 1200 --state "$RUN/state" >"$RUN/mcu-01.log" 2>&1 &
B=$!
# второй держатель весов, помощнее
./bin/dcd --role shard --port 9103 --tag board-1 --free-mem 2500 \
         --state "$RUN/state" >"$RUN/board-1.log" 2>&1 &
C=$!
trap 'kill $A $B $C 2>/dev/null || true' EXIT
sleep 1

cat > "$RUN/distcomp.conf" <<EOF
[llm]
api_key      =
timeout_ms   = 3000
payload_mode = expr

[role compute]
daemon = 127.0.0.1:9101
daemon = 127.0.0.1:9102

[role verify]
daemon = 127.0.0.1:9101

[role memorize]
daemon = 127.0.0.1:9102

[scheduler]
concurrency = 4
retries     = 1
timeout_ms  = 5000
outdir      = $RUN/runs
EOF

# устройства, между которыми раскладывается модель: у lite-узла памяти мало,
# поэтому часть слоёв достанется серверу
cat > "$RUN/shard.conf" <<EOF
[llm]
api_key =
[role shard]
daemon = 127.0.0.1:9102     # у микроконтроллера памяти мало
daemon = 127.0.0.1:9103
[scheduler]
timeout_ms = 5000
retries   = 0
outdir    = $RUN/runs
EOF

echo "=== проверка узлов: что каждый о себе сообщает ==="
./bin/dcctl -c "$RUN/distcomp.conf" check

echo
echo "=== обнаружение узлов (широковещание) ==="
./bin/dcctl -c "$RUN/distcomp.conf" discover 3 || true

echo
echo "=== назначение ролей (выбирает пользователь) ==="
./bin/dcctl -c "$RUN/distcomp.conf" roles

GOAL='посчитать стоимость закупки 1250*3 и доставку 480, затем разделить (1250*3+480)/7'

echo "=== план ==="
./bin/dcctl -c "$RUN/distcomp.conf" --no-llm plan "$GOAL"

echo "=== выполнение ==="
./bin/dcctl -c "$RUN/distcomp.conf" --no-llm run "$GOAL"

echo
echo "=== шаг не помещается в lite-узел: кого это касается ==="
python3 - <<PY
import json
json.dump({'goal': 'перевес лимита', 'steps': [
    {'id': 'small', 'role': 'compute', 'title': 'мелкий шаг', 'payload': '2+2', 'deps': []},
    {'id': 'big',   'role': 'compute', 'title': 'шаг 4 КБ', 'payload': '1+'*2000+'1', 'deps': []},
]}, open('$RUN/over.json','w'))
PY
# узел с маленьким лимитом назначаем единственным в роль — тогда некуда деться
cat > "$RUN/only-lite.conf" <<EOF
[llm]
api_key =
[role compute]
daemon = 127.0.0.1:9102
[scheduler]
retries = 0
timeout_ms = 3000
outdir = $RUN/runs
EOF
./bin/dcctl -c "$RUN/only-lite.conf" --no-llm -f "$RUN/over.json" run || true

echo
echo "=== модель, разложенная по памяти устройств ==="
# тот же lite-узел, но теперь как держатель части модели: у него 8 КБ пула
./bin/dcctl -c "$RUN/shard.conf" shard gen "$RUN/model.bin" 6 8
./bin/dcctl -c "$RUN/shard.conf" shard info "$RUN/model.bin"
./bin/dcctl -c "$RUN/shard.conf" shard deploy "$RUN/model.bin" | tail -4
echo
./bin/dcctl -c "$RUN/shard.conf" shard run "$RUN/model.bin" 0.5 -0.25 1 0.75 -1 0.3 0.2 -0.8 | tail -8

echo
echo "=== журналы узлов ==="
for f in server-a mcu-01 board-1; do
  echo "--- $f"
  grep -E '^\[' "$RUN/$f.log" | tail -6 || true
done

echo
echo "=== что запомнил узел memorize ==="
cat "$RUN/state/memory.log" 2>/dev/null || echo "(пусто)"
