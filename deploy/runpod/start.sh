#!/usr/bin/env bash
# Sobe tudo quando o pod liga (chamado pelo comando do container).
# Ordem: pacotes de sistema -> Postgres -> Redis -> S3 local -> migrations -> API (8000) -> worker -> auto-desligar.
set -uo pipefail
source /workspace/motion/common.sh
L=$M/logs; mkdir -p $L
echo "== boot $(date -u)"
apt_install
PGBIN=$(ls -d /usr/lib/postgresql/*/bin | tail -1)
set -a; source $M/env; set +a
export RUNPOD_POD_ID RUNPOD_API_KEY 2>/dev/null || true

chown -R postgres:postgres $M/pg 2>/dev/null; chmod 700 $M/pg
rm -f $M/pg/postmaster.pid      # pod parado à força deixa o pid velho
su postgres -c "$PGBIN/pg_ctl -D $M/pg -l $L/postgres.log -o '-c listen_addresses=127.0.0.1 -c fsync=off -c synchronous_commit=off' -w start"
su postgres -c "psql -tAc \"select 1 from pg_roles where rolname='motion'\"" | grep -q 1 \
  || su postgres -c "psql -c \"create user motion password 'motion'\" -c 'create database motion owner motion'"

redis-server --port 6379 --bind 127.0.0.1 --dir $M/redis --daemonize yes --logfile $L/redis.log
nohup rclone serve s3 $M/s3 --addr 127.0.0.1:9000 --auth-key motion,motion-secret --force-path-style=true \
  --vfs-cache-mode off >> $L/s3.log 2>&1 &
for i in $(seq 30); do curl -s -o /dev/null 127.0.0.1:9000 && break; sleep 1; done

cd $APP
python -m scripts.seed >> $L/seed.log 2>&1 || echo "seed falhou (veja seed.log)"
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 >> $L/api.log 2>&1 &
nohup celery -A taskqueue.celery_app worker -Q gpu --concurrency 1 --loglevel INFO >> $L/worker.log 2>&1 &
[ "${NO_IDLE_SHUTDOWN:-0}" = 1 ] || nohup python -m scripts.idle_shutdown >> $L/idle.log 2>&1 &
echo "== pronto $(date -u)"
