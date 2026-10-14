#!/usr/bin/env bash
# Запускает N узлов-шейдеров на localhost (порты 19521..) для локальной проверки.
# Использование: scripts/start_shards.sh [N] [free_mem]
set -u
cd "$(dirname "$0")/.."
N="${1:-5}"
MEM="${2:-450000}"
pkill -f 'bin/dcd' 2>/dev/null
sleep 1
for i in $(seq 1 "$N"); do
  port=$((19520 + i))
  setsid ./bin/dcd --port "$port" --role shard --free-mem "$MEM" --tag "sh$i" \
    > "/tmp/dc/sh$i.log" 2>&1 < /dev/null &
  echo $! > "/tmp/dc/sh$i.pid"
done
sleep 1
echo "запущено узлов: $(pgrep -f 'bin/dcd' | wc -l)"
