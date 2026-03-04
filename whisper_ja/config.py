"""
Configuration for Whisper Japanese finetune.
Chỉnh sửa ở đây trước khi chạy.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # === Model ===
    # tiny/base/small/medium/large-v2/large-v3/turbo
    model_size: str = "tiny"
    # Optional override: để rỗng để tự build từ model_size
    model_name: str = ""
    language: str = "ja"
    task: str = "transcribe"

    # === Dataset ===
    # ReazonSpeech sizes: tiny(8.5h/600MB), small(100h/6GB), medium(1000h/65GB)
    reazonspeech_size: str = "small"
    max_train_samples: int = 0
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

    def __post_init__(self):
        size = self.model_size.strip()
        if not size:
            raise ValueError("model_size cannot be empty")
        self.model_size = size

        model_tag = f"whisper-{self.model_size}-ja"

        if not self.model_name:
            self.model_name = f"openai/whisper-{self.model_size}"

        # Chỉ auto-đổi các giá trị mặc định đang hardcode tiny.
        # Nếu bạn set custom value trong Config/CLI thì sẽ được giữ nguyên.
        if self.output_dir == "./output/whisper-tiny-ja-lora":
            self.output_dir = f"./output/{model_tag}-lora"
        if self.merged_output_dir == "./output/whisper-tiny-ja":
            self.merged_output_dir = f"./output/{model_tag}"
        if self.wandb_project == "whisper-tiny-ja":
            self.wandb_project = model_tag
        if self.ct2_output_dir == "./output/whisper-tiny-ja-ct2":
            self.ct2_output_dir = f"./output/{model_tag}-ct2"
        if self.hub_model_id == "dungca/whisper-tiny-ja":
            self.hub_model_id = f"dungca/{model_tag}"
        if self.hub_adapter_model_id == "dungca/whisper-tiny-ja-lora":
            self.hub_adapter_model_id = f"dungca/{model_tag}-lora"
