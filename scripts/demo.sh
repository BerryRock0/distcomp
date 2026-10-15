#!/usr/bin/env bash
# Стенд на одной машине (UDP): сначала планировщик без LLM, затем настоящая
# модель llama (GGUF), разложенная по устройствам-узлам роли `shard`.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=.demo
rm -rf "$RUN"; mkdir -p "$RUN/state"

# --- планировщик: обычный узел и lite-узел (для маленьких устройств) ------
./bin/dcd --role compute --role verify --port 9101 --tag server-a \
         --state "$RUN/state" >"$RUN/server-a.log" 2>&1 &
A=$!
# lite-сборка: лимит тела 1024 байта, пул модели 8 КБ — как у микроконтроллера
./bin/dcd-lite --role compute --role memorize --port 9102 --tag mcu-01 \
         --free-mem 1200 --state "$RUN/state" >"$RUN/mcu-01.log" 2>&1 &
B=$!

# --- устройства, между которыми раскладываем модель ----------------------
# Конвейеру нужны минимум три узла: вход, слои и выход живут на разных
# устройствах (узел держит ровно один контекст движка). Возьмём пять и дадим
# большим узлам больше памяти — им достанется больше слоёв.
./bin/dcd --role shard --port 9111 --tag in  --free-mem 450000 --state "$RUN/state" >"$RUN/sh-in.log"  2>&1 &
S1=$!
./bin/dcd --role shard --port 9112 --tag big --free-mem 450000 --state "$RUN/state" >"$RUN/sh-big.log" 2>&1 &
S2=$!
./bin/dcd --role shard --port 9113 --tag sm1 --free-mem 250000 --state "$RUN/state" >"$RUN/sh-sm1.log" 2>&1 &
S3=$!
./bin/dcd --role shard --port 9114 --tag sm2 --free-mem 250000 --state "$RUN/state" >"$RUN/sh-sm2.log" 2>&1 &
S4=$!
./bin/dcd --role shard --port 9115 --tag out --free-mem 450000 --state "$RUN/state" >"$RUN/sh-out.log" 2>&1 &
S5=$!
trap 'kill $A $B $S1 $S2 $S3 $S4 $S5 2>/dev/null || true' EXIT
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

# устройства для раскладки модели (роль shard)
cat > "$RUN/shard.conf" <<EOF
[llm]
api_key =
[role shard]
daemon = 127.0.0.1:9111
daemon = 127.0.0.1:9112
daemon = 127.0.0.1:9113
daemon = 127.0.0.1:9114
daemon = 127.0.0.1:9115
[scheduler]
timeout_ms = 8000
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
echo "=== модель llama (GGUF), разложенная по памяти устройств ==="
# эталонная мини-модель со всеми деталями настоящей: RMSNorm, RoPE,
# grouped-query attention, SwiGLU, KV-кэш, BPE-токенизатор
python3 scripts/llama_ref.py "$RUN/model.gguf" --json "$RUN/ref.json"
echo
./bin/dcctl -c "$RUN/shard.conf" llm info "$RUN/model.gguf"
./bin/dcctl -c "$RUN/shard.conf" llm deploy "$RUN/model.gguf" | tail -6
echo
./bin/dcctl -c "$RUN/shard.conf" llm prompt "$RUN/model.gguf" "distcomp" 8 | tail -6
echo
echo "=== независимый эталон на numpy: те же жадные токены ==="
python3 scripts/gguf_expected.py --prompt distcomp --tokens 8

echo
echo "=== журналы узлов ==="
for f in server-a mcu-01; do
  echo "--- $f"
  grep -E '^\[' "$RUN/$f.log" | tail -6 || true
done

echo
echo "=== что запомнил узел memorize ==="
cat "$RUN/state/memory.log" 2>/dev/null || echo "(пусто)"
