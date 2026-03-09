#!/bin/bash
# =============================================================================
# vast_setup.sh — Bootstrap Vast.ai instance cho Whisper Japanese ASR
#
# Chạy ngay sau khi SSH vào instance lần đầu:
#   bash vast_setup.sh
#
# Hoặc truyền GitHub repo:
#   GITHUB_REPO=dungca1512/whisper-finetune-ja bash vast_setup.sh
# =============================================================================

set -e

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_REPO="${GITHUB_REPO:-dungca1512/whisper-finetune-ja}"
WORK_DIR="${WORK_DIR:-/workspace/whisper-finetune-ja}"
PYTHON="${PYTHON:-python3}"

echo "============================================================"
echo " Vast.ai Bootstrap — Whisper Japanese ASR"
echo " Repo:     https://github.com/${GITHUB_REPO}"
echo " Work dir: ${WORK_DIR}"
echo "============================================================"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    git \
    curl \
    wget \
    vim \
    tmux \
    htop \
    ffmpeg \
    libsndfile1 \
    build-essential
echo "✅ System packages done"
echo ""

# ── 2. Verify GPU ─────────────────────────────────────────────────────────────
echo "[2/6] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "✅ GPU detected"
else
    echo "⚠️  nvidia-smi not found — no GPU or driver not installed"
fi
echo ""

# ── 3. Clone repo ─────────────────────────────────────────────────────────────
echo "[3/6] Cloning repo..."
if [ -d "${WORK_DIR}" ]; then
    echo "   Directory exists, pulling latest..."
    git -C "${WORK_DIR}" pull
else
    git clone "https://github.com/${GITHUB_REPO}.git" "${WORK_DIR}"
fi
cd "${WORK_DIR}"
echo "✅ Repo ready at ${WORK_DIR}"
echo ""

# ── 4. Python dependencies ────────────────────────────────────────────────────
echo "[4/6] Installing Python dependencies..."
${PYTHON} -m pip install --upgrade pip -q

# Vast.ai PyTorch template đã có torch+CUDA sẵn
# Chỉ install thêm project deps, skip torch để tránh overwrite
${PYTHON} -m pip install \
    transformers>=4.40 \
    datasets==2.21.0 \
    accelerate \
    evaluate \
    jiwer \
    soundfile \
    librosa \
    wandb \
    huggingface_hub \
    peft \
    ctranslate2 \
    faster-whisper \
    tensorboard \
    python-dotenv \
    optimum[exporters] \
    onnxruntime-gpu \
    -q

echo "✅ Python dependencies done"
echo ""

# ── 5. Verify torch + CUDA ────────────────────────────────────────────────────
echo "[5/6] Verifying torch + CUDA..."
${PYTHON} - <<'PY'
import torch
print(f"   torch version : {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU           : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PY
echo "✅ Torch check done"
echo ""

# ── 6. Setup .env ─────────────────────────────────────────────────────────────
echo "[6/6] Setting up .env..."
cd "${WORK_DIR}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "   ⚠️  .env created from .env.example"
    echo "   👉 Bạn cần điền secrets vào .env:"
    echo ""
    echo "      vim ${WORK_DIR}/.env"
    echo ""
    echo "   Các fields cần điền:"
    echo "      HF_TOKEN=hf_xxxx"
    echo "      WANDB_API_KEY=xxxx"
else
    echo "   .env already exists, skipping"
fi
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " ✅ Setup complete!"
echo "============================================================"
echo ""
echo " Work dir: ${WORK_DIR}"
echo ""
echo " Bước tiếp theo:"
echo ""
echo " 1) Điền HF_TOKEN vào .env:"
echo "    vim ${WORK_DIR}/.env"
echo ""
echo " 2) Train whisper-small với LoRA:"
echo "    cd ${WORK_DIR}"
echo "    tmux new -s train"
echo "    python train.py \\"
echo "        --model_size small \\"
echo "        --reazonspeech_size small \\"
echo "        --batch_size 16 \\"
echo "        --num_train_epochs 3 \\"
echo "        --wandb_tags vast,lora,reazonspeech-small"
echo ""
echo " 3) Sau khi train xong, export ONNX:"
echo "    python serving.py --export_onnx --model_size small"
echo ""
echo " 4) Build TRT engine:"
echo "    python serving.py --export_tensorrt --model_size small"
echo ""
echo " 5) Generate Triton config:"
echo "    python serving.py --gen_triton --model_size small"
echo ""
echo " 💡 Tip: Dùng tmux để session không bị mất khi SSH disconnect:"
echo "    tmux new -s train      # tạo session"
echo "    tmux attach -t train   # attach lại sau khi reconnect"
echo "============================================================"
