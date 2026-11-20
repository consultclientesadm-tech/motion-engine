#!/usr/bin/env bash
# Instalação ÚNICA no Network Volume (rode uma vez num pod com o volume em /workspace).
# Idempotente: pode rodar de novo para completar o que faltou.
set -euo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p $M/{bin,models,logs,s3/motion-engine,redis,hf,u2net,torch,ultralytics}
apt_install
PGBIN=$(ls -d /usr/lib/postgresql/*/bin | tail -1)

# Redis 8 (a imagem/apt traz 6.0; o kombu atual precisa de 6.2+)
if [ ! -x $M/bin/redis-server ]; then
  cd /tmp && wget -q https://download.redis.io/releases/redis-8.0.2.tar.gz && tar xzf redis-8.0.2.tar.gz
  cd redis-8.0.2 && make -j"$(nproc)" BUILD_TLS=no >/dev/null && cp src/redis-server src/redis-cli $M/bin/
fi
# S3 local: rclone serve s3 grava arquivos comuns no volume (o MinIO não distribui mais o binário aberto)
if [ ! -x $M/bin/rclone ]; then
  cd /tmp && wget -q https://downloads.rclone.org/v1.71.1/rclone-v1.71.1-linux-amd64.zip \
    && python -c "import zipfile;zipfile.ZipFile('rclone-v1.71.1-linux-amd64.zip').extractall('.')" \
    && cp rclone-v1.71.1-linux-amd64/rclone $M/bin/ && chmod +x $M/bin/rclone
fi
# Blender 4.2 LTS
if [ ! -x $M/blender/blender ]; then
  cd /tmp && wget -q https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz \
    && tar -xf blender-4.2.3-linux-x64.tar.xz && rm -rf $M/blender && mv blender-4.2.3-linux-x64 $M/blender
fi
ln -sf $M/blender/blender $M/bin/blender

# Python: venv que reaproveita o torch 2.4.1+cu124 da imagem (não ocupa o volume)
[ -x $M/venv/bin/python ] || python3.11 -m venv --system-site-packages $M/venv
printf "torch==2.4.1\ntorchvision==0.19.1\ntorchaudio==2.4.1\nnumpy<2\n" > $M/constraints.txt
cd $APP
pip install -q -c $M/constraints.txt -r requirements-worker.txt pytest moto
TP=$(python -c "import torch.utils; print(torch.utils.cmake_prefix_path)")
# sem CMAKE_PREFIX_PATH o CMake do torchmcubes não acha o Torch da imagem
pip show torchmcubes >/dev/null 2>&1 || CMAKE_PREFIX_PATH=$TP Torch_DIR=$TP/Torch CUDACXX=/usr/local/cuda/bin/nvcc pip install -q --no-build-isolation git+https://github.com/tatsy/torchmcubes.git
pip uninstall -y -q onnxruntime onnxruntime-gpu || true
pip install -q --no-deps "onnxruntime-gpu>=1.18,<1.23"
BS=$(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('basicsr').origin))")
sed -i 's/from torchvision.transforms.functional_tensor import/from torchvision.transforms.functional import/' $BS/data/degradations.py
[ -d $M/TripoSR ] || git clone -q --depth 1 https://github.com/VAST-AI-Research/TripoSR $M/TripoSR
[ -d $M/Wav2Lip ] || git clone -q --depth 1 https://github.com/Rudrabha/Wav2Lip $M/Wav2Lip

# Pesos
bash scripts/download_models.sh $M/models || true
for w in yolov8m.pt yolov8s-seg.pt; do
  [ -s $M/models/$w ] || wget -q -O $M/models/$w https://github.com/ultralytics/assets/releases/download/v8.3.0/$w
done
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/TripoSR")
try:
    import rembg; rembg.new_session("u2net")
except Exception as e: print("rembg:", e)
PY

# Banco (uma vez)
if [ ! -s $M/pg/PG_VERSION ]; then
  mkdir -p $M/pg && chown postgres:postgres $M/pg && chmod 700 $M/pg
  su postgres -c "$PGBIN/initdb -D $M/pg -A trust -U postgres" >/dev/null
fi
# .env do motor (fora de app/ para um rsync do código não apagar)
[ -s $M/env ] || cat > $M/env <<ENV
DATABASE_URL=postgresql+psycopg://motion:motion@127.0.0.1:5432/motion
REDIS_URL=redis://127.0.0.1:6379/0
S3_ENDPOINT=http://127.0.0.1:9000
S3_BUCKET=motion-engine
S3_ACCESS_KEY=motion
S3_SECRET_KEY=motion-secret
S3_REGION=us-east-1
WORK_DIR=/tmp/motion-engine
BLENDER_BIN=$M/blender/blender
YOLO_WEIGHTS=$M/models/yolov8m.pt
YOLO_SEG_WEIGHTS=$M/models/yolov8s-seg.pt
LAMA_WEIGHTS=$M/models/big-lama.pt
DWPOSE_DET_ONNX=$M/models/yolox_l.onnx
DWPOSE_POSE_ONNX=$M/models/dw-ll_ucoco_384.onnx
REALESRGAN_WEIGHTS=$M/models/RealESRGAN_x2plus.pth
WAV2LIP_CHECKPOINT=$M/models/wav2lip_gan.pth
WAV2LIP_REPO=$M/Wav2Lip
GPU_USD_PER_HOUR=0.59
ENV
cp "$(dirname "$0")/start.sh" "$(dirname "$0")/common.sh" $M/
echo "setup ok"
