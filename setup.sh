#!/bin/bash
# =============================================================================
# Setup script cho Vast.ai
# Chạy: bash setup.sh
# =============================================================================

set -e

echo "🔧 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Usage:"
echo "  # Train LoRA với ReazonSpeech small (100h)"
echo "  python train.py --reazonspeech_size small --batch_size 32 --no_wandb"
echo ""
echo "  # Train với custom LoRA settings"
echo "  python train.py --reazonspeech_size small --batch_size 32 --num_train_epochs 5 --learning_rate 5e-6 --lora_r 16 --lora_alpha 32"
echo ""
echo "  # Train không có W&B"
echo "  python train.py --reazonspeech_size small --no_wandb"
echo ""
echo "  # Export model đã train"
echo "  python train.py --export_only"
echo ""
echo "  # Test inference"
echo "  python train.py --test_only"
echo ""
echo "  # Push adapter + merged model lên Hub"
echo "  python train.py --push_to_hub --hub_adapter_model_id yourname/whisper-tiny-ja-lora --hub_model_id yourname/whisper-tiny-ja --no_wandb"
echo ""
echo "Đừng quên set environment variables:"
echo "  export HF_TOKEN=hf_xxxxx"
echo "  export WANDB_API_KEY=xxxxx"
