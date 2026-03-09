"""
Triton Inference Server config generator for Whisper.

Generates model_repository layout with config.pbtxt files
for encoder and decoder TensorRT engines.

Supports:
  - TensorRT backend (GPU)
  - ONNX Runtime backend (CPU fallback)
  - Ensemble pipeline (encoder → decoder)

Usage:
    from whisper_ja.serving.triton_config import generate_triton_repository
    generate_triton_repository(config, trt_dir="./output/whisper-small-ja-trt")
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Config.pbtxt templates
# ---------------------------------------------------------------------------

_ENCODER_TRT_CONFIG = """\
name: "whisper_encoder"
backend: "tensorrt"
max_batch_size: 4

input [
  {{
    name: "input_features"
    data_type: TYPE_FP16
    dims: [ 80, 3000 ]
  }}
]

output [
  {{
    name: "last_hidden_state"
    data_type: TYPE_FP16
    dims: [ -1, 1500 ]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 1, 2, 4 ]
  max_queue_delay_microseconds: 5000
}}
"""

_DECODER_TRT_CONFIG = """\
name: "whisper_decoder"
backend: "tensorrt"
max_batch_size: 4

input [
  {{
    name: "encoder_hidden_states"
    data_type: TYPE_FP16
    dims: [ -1, 1500 ]
  }},
  {{
    name: "decoder_input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP16
    dims: [ -1, 51865 ]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
"""

_ENCODER_ONNX_CONFIG = """\
name: "whisper_encoder"
backend: "onnxruntime"
max_batch_size: 4

input [
  {{
    name: "input_features"
    data_type: TYPE_FP32
    dims: [ 80, 3000 ]
  }}
]

output [
  {{
    name: "last_hidden_state"
    data_type: TYPE_FP32
    dims: [ -1, 1500 ]
  }}
]

instance_group [
  {{
    kind: KIND_CPU
    count: 1
  }}
]
"""

_DECODER_ONNX_CONFIG = """\
name: "whisper_decoder"
backend: "onnxruntime"
max_batch_size: 4

input [
  {{
    name: "encoder_hidden_states"
    data_type: TYPE_FP32
    dims: [ -1, 1500 ]
  }},
  {{
    name: "decoder_input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, 51865 ]
  }}
]

instance_group [
  {{
    kind: KIND_CPU
    count: 1
  }}
]
"""

_ENSEMBLE_CONFIG = """\
name: "whisper_pipeline"
platform: "ensemble"
max_batch_size: 4

input [
  {{
    name: "input_features"
    data_type: TYPE_FP32
    dims: [ 80, 3000 ]
  }}
]

output [
  {{
    name: "transcription_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]

ensemble_scheduling {{
  step [
    {{
      model_name: "whisper_encoder"
      model_version: 1
      input_map {{
        key: "input_features"
        value: "input_features"
      }}
      output_map {{
        key: "last_hidden_state"
        value: "encoder_output"
      }}
    }},
    {{
      model_name: "whisper_decoder"
      model_version: 1
      input_map {{
        key: "encoder_hidden_states"
        value: "encoder_output"
      }}
      input_map {{
        key: "decoder_input_ids"
        value: "decoder_input_ids"
      }}
      output_map {{
        key: "logits"
        value: "transcription_ids"
      }}
    }}
  ]
}}
"""

_DOCKER_COMPOSE = """\
version: "3.8"

services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.01-py3
    runtime: nvidia
    ports:
      - "8000:8000"   # HTTP
      - "8001:8001"   # gRPC
      - "8002:8002"   # Metrics
    volumes:
      - ./model_repository:/models
    command: >
      tritonserver
        --model-repository=/models
        --log-verbose=1
        --strict-model-config=false
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # CPU-only variant (ONNX backend, no GPU required)
  triton-cpu:
    image: nvcr.io/nvidia/tritonserver:24.01-py3-cpu
    ports:
      - "8010:8000"
      - "8011:8001"
      - "8012:8002"
    volumes:
      - ./model_repository_cpu:/models
    command: >
      tritonserver
        --model-repository=/models
        --backend-config=onnxruntime,enable_mem_arena=0
        --log-verbose=1
"""

_CLIENT_EXAMPLE = '''\
#!/usr/bin/env python3
"""
Example Triton client for Whisper ASR inference.

Install: pip install tritonclient[all]

Usage:
    python triton_client_example.py --audio ./audio.wav --server localhost:8001
"""

from __future__ import annotations

import argparse
import numpy as np
import soundfile as sf


def transcribe_with_triton(audio_path: str, server_url: str = "localhost:8001") -> str:
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        raise RuntimeError("Install tritonclient: pip install tritonclient[grpc]")

    try:
        from transformers import WhisperFeatureExtractor
    except ImportError:
        raise RuntimeError("Install transformers: pip install transformers")

    audio, sr = sf.read(audio_path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    features = feature_extractor(
        audio, sampling_rate=16000, return_tensors="np"
    ).input_features  # shape: (1, 80, 3000)

    client = grpcclient.InferenceServerClient(url=server_url)

    inputs = [grpcclient.InferInput("input_features", features.shape, "FP32")]
    inputs[0].set_data_from_numpy(features.astype(np.float32))

    outputs = [grpcclient.InferRequestedOutput("transcription_ids")]
    response = client.infer(model_name="whisper_pipeline", inputs=inputs, outputs=outputs)

    token_ids = response.as_numpy("transcription_ids")
    print(f"Raw token ids: {token_ids}")
    return str(token_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--server", default="localhost:8001")
    args = parser.parse_args()
    result = transcribe_with_triton(args.audio, args.server)
    print(f"Result: {result}")
'''


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_triton_repository(
    config,
    trt_dir: str | None = None,
    onnx_dir: str | None = None,
    output_dir: str | None = None,
    backend: str = "auto",
) -> dict:
    """
    Generate Triton model_repository layout.

    Args:
        config: Config instance
        trt_dir: TensorRT engine directory (for GPU backend)
        onnx_dir: ONNX model directory (for CPU backend)
        output_dir: Where to write model_repository. Defaults to ./output/triton
        backend: "tensorrt" | "onnxruntime" | "auto" (auto-detects from available files)

    Returns:
        dict with generation result.
    """
    resolved_trt_dir = trt_dir or getattr(
        config, "tensorrt_output_dir",
        f"./output/whisper-{config.model_size}-ja-trt"
    )
    resolved_onnx_dir = onnx_dir or getattr(
        config, "onnx_output_dir",
        f"./output/whisper-{config.model_size}-ja-onnx"
    )
    triton_dir = output_dir or getattr(
        config, "triton_output_dir",
        f"./output/triton-whisper-{config.model_size}"
    )

    # Auto-detect backend
    if backend == "auto":
        trt_path = Path(resolved_trt_dir)
        has_trt = trt_path.exists() and any(trt_path.glob("*.plan"))
        backend = "tensorrt" if has_trt else "onnxruntime"

    print(f"🔧 Generating Triton model repository")
    print(f"   Backend: {backend}")
    print(f"   Output:  {triton_dir}")

    repo_root = Path(triton_dir)
    repo_root.mkdir(parents=True, exist_ok=True)

    # 1. Encoder
    encoder_dir = repo_root / "whisper_encoder" / "1"
    encoder_dir.mkdir(parents=True, exist_ok=True)

    if backend == "tensorrt":
        encoder_engine = Path(resolved_trt_dir) / "encoder.plan"
        if encoder_engine.exists():
            shutil.copy2(encoder_engine, encoder_dir / "model.plan")
        (repo_root / "whisper_encoder" / "config.pbtxt").write_text(
            _ENCODER_TRT_CONFIG, encoding="utf-8"
        )
    else:
        encoder_onnx = Path(resolved_onnx_dir) / "encoder_model.onnx"
        if encoder_onnx.exists():
            shutil.copy2(encoder_onnx, encoder_dir / "model.onnx")
        (repo_root / "whisper_encoder" / "config.pbtxt").write_text(
            _ENCODER_ONNX_CONFIG, encoding="utf-8"
        )

    # 2. Decoder
    decoder_dir = repo_root / "whisper_decoder" / "1"
    decoder_dir.mkdir(parents=True, exist_ok=True)

    if backend == "tensorrt":
        decoder_engine = Path(resolved_trt_dir) / "decoder_merged.plan"
        if not decoder_engine.exists():
            decoder_engine = Path(resolved_trt_dir) / "decoder.plan"
        if decoder_engine.exists():
            shutil.copy2(decoder_engine, decoder_dir / "model.plan")
        (repo_root / "whisper_decoder" / "config.pbtxt").write_text(
            _DECODER_TRT_CONFIG, encoding="utf-8"
        )
    else:
        decoder_onnx = Path(resolved_onnx_dir) / "decoder_model_merged.onnx"
        if not decoder_onnx.exists():
            decoder_onnx = Path(resolved_onnx_dir) / "decoder_model.onnx"
        if decoder_onnx.exists():
            shutil.copy2(decoder_onnx, decoder_dir / "model.onnx")
        (repo_root / "whisper_decoder" / "config.pbtxt").write_text(
            _DECODER_ONNX_CONFIG, encoding="utf-8"
        )

    # 3. Ensemble pipeline
    ensemble_dir = repo_root / "whisper_pipeline" / "1"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / "whisper_pipeline" / "config.pbtxt").write_text(
        _ENSEMBLE_CONFIG, encoding="utf-8"
    )

    # 4. Docker Compose
    (repo_root / "docker-compose.yml").write_text(_DOCKER_COMPOSE, encoding="utf-8")

    # 5. Client example
    (repo_root / "triton_client_example.py").write_text(_CLIENT_EXAMPLE, encoding="utf-8")

    # 6. Metadata
    meta = {
        "model_size": config.model_size,
        "backend": backend,
        "trt_dir": resolved_trt_dir,
        "onnx_dir": resolved_onnx_dir,
        "triton_repo": triton_dir,
        "models": ["whisper_encoder", "whisper_decoder", "whisper_pipeline"],
    }
    (repo_root / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"✅ Triton repository generated: {triton_dir}")
    print(f"   Models: whisper_encoder, whisper_decoder, whisper_pipeline (ensemble)")
    print(f"\nTo start Triton server:")
    print(f"   cd {triton_dir} && docker compose up triton")
    print(f"\nFor CPU-only (ONNX backend):")
    print(f"   cd {triton_dir} && docker compose up triton-cpu")

    return {
        "success": True,
        "backend": backend,
        "triton_repo": triton_dir,
        "meta": meta,
    }
