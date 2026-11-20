# Variáveis comuns aos scripts do pod (setup.sh e start.sh).
M=/workspace/motion            # tudo que persiste fica no Network Volume
APP=$M/app
export PATH=$M/bin:$M/venv/bin:$PATH
export HF_HOME=$M/hf U2NET_HOME=$M/u2net TORCH_HOME=$M/torch YOLO_CONFIG_DIR=$M/ultralytics
export PYTHONPATH=$M/TripoSR:$APP
export PYTHONUNBUFFERED=1
# pacotes de sistema (não persistem: o disco do container zera a cada boot)
APT_PKGS="postgresql libxi6 libxrender1 libxkbcommon0 libsm6 libegl1 libgl1 libglib2.0-0 ffmpeg rsync"
apt_install() {
  # o usuário do postgres com uid fixo: os arquivos do banco no volume continuam dele em todo boot
  id postgres >/dev/null 2>&1 || { groupadd -g 7777 postgres; useradd -r -u 7777 -g 7777 -d /var/lib/postgresql -s /bin/bash postgres; }
  local missing=0
  for p in postgresql ffmpeg libegl1; do dpkg -s $p >/dev/null 2>&1 || missing=1; done
  if [ $missing = 1 ]; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $APT_PKGS >/dev/null
  fi
}
PGBIN=$(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | tail -1 || true)
