"""
Configuration for Whisper Japanese finetune.
Chỉnh sửa ở đây trước khi chạy.
"""

from dataclasses import dataclass, field


DEFAULT_MODEL_SIZE = "small"
DEFAULT_REAZONSPEECH_SIZE = "small"
DEFAULT_HF_REPO_OWNER = "dungca"


@dataclass
class Config:
    # === Model ===
    # tiny/base/small/medium/large-v2/large-v3/turbo
    model_size: str = DEFAULT_MODEL_SIZE
    # Optional override: để rỗng để tự build từ model_size
    model_name: str = ""
    language: str = "ja"
    task: str = "transcribe"

    # === Dataset ===
    # ReazonSpeech sizes: tiny(8.5h/600MB), small(100h/6GB), medium(1000h/65GB)
    reazonspeech_size: str = DEFAULT_REAZONSPEECH_SIZE
    max_train_samples: int = 0
    eval_dataset_name: str = "japanese-asr/ja_asr.reazonspeech_test"
    max_eval_samples: int = 2000

    # === HuggingFace ===
    hf_token: str = ""  # Keep empty; read from env/secret HF_TOKEN at runtime.
    hf_repo_owner: str = DEFAULT_HF_REPO_OWNER

    # === Training ===
    # output_dir lưu adapter LoRA (nhẹ, nhanh upload/checkpoint)
    output_dir: str = ""
    # merged_output_dir lưu full weights sau khi merge LoRA để deploy/export
    merged_output_dir: str = ""
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
    wandb_project: str = ""
    wandb_key: str = ""  # Keep empty; read from env/secret WANDB_API_KEY at runtime.
    wandb_tags: list[str] = field(default_factory=list)

    # === Preprocessing ===
    num_proc: int = 8  # Số CPU cores dùng cho preprocessing

    # === Export ===
    export_ct2: bool = True
    ct2_output_dir: str = ""
    ct2_quantization: str = "int8"

    # === Push to Hub ===
    push_to_hub: bool = False
    push_merged_to_hub: bool = True
    hub_model_id: str = ""
    hub_adapter_model_id: str = ""

    def __post_init__(self):
        size = self.model_size.strip()
        if not size:
            raise ValueError("model_size cannot be empty")
        self.model_size = size

        repo_owner = self.hf_repo_owner.strip()
        if not repo_owner:
            repo_owner = DEFAULT_HF_REPO_OWNER
        self.hf_repo_owner = repo_owner

        model_tag = f"whisper-{self.model_size}-ja"

        legacy_model_name = f"openai/whisper-{DEFAULT_MODEL_SIZE}"
        if not self.model_name or self.model_name == legacy_model_name:
            self.model_name = f"openai/whisper-{self.model_size}"

        legacy_output_dir = f"./output/whisper-{DEFAULT_MODEL_SIZE}-ja-lora"
        if not self.output_dir or self.output_dir == legacy_output_dir:
            self.output_dir = f"./output/{model_tag}-lora"

        legacy_merged_dir = f"./output/whisper-{DEFAULT_MODEL_SIZE}-ja"
        if not self.merged_output_dir or self.merged_output_dir == legacy_merged_dir:
            self.merged_output_dir = f"./output/{model_tag}"

        legacy_wandb_project = f"whisper-{DEFAULT_MODEL_SIZE}-ja"
        if not self.wandb_project or self.wandb_project == legacy_wandb_project:
            self.wandb_project = model_tag

        legacy_ct2_dir = f"./output/whisper-{DEFAULT_MODEL_SIZE}-ja-ct2"
        if not self.ct2_output_dir or self.ct2_output_dir == legacy_ct2_dir:
            self.ct2_output_dir = f"./output/{model_tag}-ct2"

        legacy_hub_model_id = f"{DEFAULT_HF_REPO_OWNER}/whisper-{DEFAULT_MODEL_SIZE}-ja"
        if not self.hub_model_id or self.hub_model_id == legacy_hub_model_id:
            self.hub_model_id = f"{self.hf_repo_owner}/{model_tag}"

        legacy_hub_adapter_id = f"{DEFAULT_HF_REPO_OWNER}/whisper-{DEFAULT_MODEL_SIZE}-ja-lora"
        if not self.hub_adapter_model_id or self.hub_adapter_model_id == legacy_hub_adapter_id:
            self.hub_adapter_model_id = f"{self.hf_repo_owner}/{model_tag}-lora"
