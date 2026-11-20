#!/usr/bin/env bash
# Baixa os pesos usados pelos workers para a pasta indicada (padrão: models/).
# YOLOv8 e TripoSR são baixados sozinhos na primeira execução.
set -euo pipefail
DIR=${1:-models}
mkdir -p "$DIR"
get() { [ -s "$DIR/$2" ] || wget -q --show-progress -O "$DIR/$2" "$1"; }
get https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx dw-ll_ucoco_384.onnx
get https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx yolox_l.onnx
get https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth RealESRGAN_x2plus.pth
if [ ! -s "$DIR/wav2lip_gan.pth" ]; then
  echo "AVISO: baixe wav2lip_gan.pth manualmente (link no README do Wav2Lip) para $DIR/" >&2
fi
# LaMa (TorchScript) para preencher o fundo onde a pessoa original estava
get https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt big-lama.pt
