"""
Export finetuned model to faster-whisper (CTranslate2) format.
"""

import subprocess
import os


def export_to_ct2(config, model_dir=None):
    """Convert HF model → CTranslate2 INT8 for faster-whisper."""
    if not config.export_ct2:
        print("⏭️  Skipping CT2 export (disabled in config)")
        return

    source_model_dir = model_dir or config.output_dir
    if not os.path.isdir(source_model_dir):
        print(f"❌ Source model directory not found for CT2 export: {source_model_dir}")
        return
    if (
        os.path.isfile(os.path.join(source_model_dir, "adapter_config.json")) and
        not os.path.isfile(os.path.join(source_model_dir, "model.safetensors")) and
        not os.path.isfile(os.path.join(source_model_dir, "pytorch_model.bin"))
    ):
        print(
            "❌ Source directory contains only LoRA adapter weights. "
            "Merge adapter first or point --model_dir to merged full model."
        )
        return

    print(f"📦 Exporting to CTranslate2 ({config.ct2_quantization})...")
    print(f"   Source model: {source_model_dir}")

    os.makedirs(config.ct2_output_dir, exist_ok=True)

    cmd = [
        "ct2-whisper-converter",
        "--model", source_model_dir,
        "--output_dir", config.ct2_output_dir,
        "--quantization", config.ct2_quantization,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        print(f"✅ Exported to: {config.ct2_output_dir}")
        print()
        print("Sử dụng:")
        print('  from faster_whisper import WhisperModel')
        print(f'  model = WhisperModel("{config.ct2_output_dir}", device="cpu", compute_type="{config.ct2_quantization}")')
        print('  segments, info = model.transcribe("audio.wav", language="ja")')
    except FileNotFoundError:
        print("❌ ct2-whisper-converter not found. Install: pip install ctranslate2")
    except subprocess.CalledProcessError as e:
        print(f"❌ Export failed: {e.stderr}")
