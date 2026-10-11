#!/usr/bin/env bash
# UAIA (U ask, I answer) v1: start the trained judge (if it is not running) and the web page.
# Usage: ./run_v1.sh        Stop with Ctrl+C.
#        JUDGE_DEVICE=cuda ./run_v1.sh   chooses the judge's device when this script starts it
#                                        (cuda, mps or cpu; default: auto).
set -euo pipefail
cd "$(dirname "$0")"

JUDGE_PY="model selection/.colab-venv/bin/python"
APP_PY=".venv/bin/python"
ADAPTER="model selection/models/judge_adapter"
JUDGE_URL="http://127.0.0.1:8765"
APP_PORT=8000
EXTRA=()
mkdir -p data/private

if [ ! -f "$ADAPTER/adapter_config.json" ]; then
  echo "No judge adapter in '$ADAPTER'."
  echo "Download checkpoint_0150/adapter from Drive, rename it judge_adapter and put it in model selection/models/."
  echo "Continuing without the judge (rule-based tie-break)."
  EXTRA=(--no-judge)
elif curl -s -m 2 "$JUDGE_URL/health" >/dev/null; then
  echo "Judge already running."
else
  echo "Starting the judge; the first load takes about a minute (log: data/private/judge_server.log)..."
  "$JUDGE_PY" -m judge_server --device "${JUDGE_DEVICE:-auto}" > data/private/judge_server.log 2>&1 &
  JUDGE_PID=$!
  trap 'kill "$JUDGE_PID" 2>/dev/null || true' EXIT
  for _ in $(seq 1 180); do
    if curl -s -m 2 "$JUDGE_URL/health" >/dev/null; then echo "Judge ready."; break; fi
    if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
      echo "The judge stopped while loading; last lines of its log:"; tail -5 data/private/judge_server.log
      EXTRA=(--no-judge); break
    fi
    sleep 2
  done
fi

( sleep 2; open "http://127.0.0.1:$APP_PORT" || xdg-open "http://127.0.0.1:$APP_PORT" ) >/dev/null 2>&1 &
"$APP_PY" -m webapp --port "$APP_PORT" "${EXTRA[@]+"${EXTRA[@]}"}"
