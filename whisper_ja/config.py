"""
Configuration for Whisper Tiny Japanese finetune.
Chỉnh sửa ở đây trước khi chạy.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # === Model ===
    model_name: str = "openai/whisper-tiny"
    language: str = "ja"
    task: str = "transcribe"

    # === Dataset ===
    # ReazonSpeech sizes: tiny(8.5h/600MB), small(100h/6GB), medium(1000h/65GB)
    reazonspeech_size: str = "small"
    eval_dataset_name: str = "japanese-asr/ja_asr.reazonspeech_test"
    max_eval_samples: int = 2000

    # === HuggingFace ===
    hf_token: str = ""  # Paste token hoặc set env HF_TOKEN

    # === Training ===
    # output_dir lưu adapter LoRA (nhẹ, nhanh upload/checkpoint)
    output_dir: str = "./output/whisper-tiny-ja-lora"
    # merged_output_dir lưu full weights sau khi merge LoRA để deploy/export
    merged_output_dir: str = "./output/whisper-tiny-ja"
    save_merged_model: bool = True
    num_train_epochs: int = 3
    batch_size: int = 64           # RTX 3090/4090: 64, T4: 32
    eval_batch_size: int = 16
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1

    # === LoRA (khuyến nghị) ===
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # === Eval & Save ===
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3
    early_stopping_patience: int = 3
    run_post_train_test: bool = True

    # === W&B ===
    use_wandb: bool = True
    wandb_project: str = "whisper-tiny-ja"
    wandb_key: str = ""  # Paste key hoặc set env WANDB_API_KEY
    wandb_tags: list[str] = field(default_factory=list)

    # === Preprocessing ===
    num_proc: int = 8  # Số CPU cores dùng cho preprocessing

    # === Export ===
    export_ct2: bool = True
    ct2_output_dir: str = "./output/whisper-tiny-ja-ct2"
    ct2_quantization: str = "int8"

    # === Push to Hub ===
    push_to_hub: bool = False
    push_merged_to_hub: bool = True
    hub_model_id: str = "dungca/whisper-tiny-ja"
    hub_adapter_model_id: str = "dungca/whisper-tiny-ja-lora"
