#!/usr/bin/env bash
# Safety net: remove our Cosmos container if system MemAvailable drops below $1 GB (protects tsf-llm and neighbours).
LIM=${1:-5}; NAME=${2:-tsf-cosmos-transfer}
while true; do
  a=$(awk "/MemAvailable/{printf \"%d\", \$2/1048576}" /proc/meminfo)
  if [ "$a" -lt "$LIM" ]; then echo "$(date -Is) MemAvailable=${a}GB < ${LIM}GB: removing $NAME" >> ~/tsf/logs/cosmos_gen_watchdog.log; docker rm -f "$NAME" >/dev/null 2>&1; fi
  sleep 2
done
