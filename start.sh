#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

# If running on RunPod Serverless, start the handler; else run FastAPI
if [[ "${RUNPOD_SERVERLESS:-}" != "" || "${RUNPOD_POD_ID:-}" != "" ]]; then
  python3 rp_handler.py | cat
else
  exec uvicorn app:app --host 0.0.0.0 --port 7861 --no-access-log
fi


