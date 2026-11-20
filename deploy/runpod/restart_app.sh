#!/usr/bin/env bash
# Reinicia só a API e o worker (depois de atualizar o código em app/).
source /workspace/motion/common.sh
set -a; source /workspace/motion/env; set +a
pkill -f "[u]vicorn api.main"; pkill -f "[c]elery -A taskqueue"; sleep 3
cd $APP
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 >> $M/logs/api.log 2>&1 &
nohup celery -A taskqueue.celery_app worker -Q gpu --concurrency 1 --loglevel INFO >> $M/logs/worker.log 2>&1 &
