"""
Model loading, LoRA setup, data collator, and metrics.
"""

import os
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor,
)


def _count_total_params(model) -> int:
    return sum(param.numel() for param in model.parameters())


def _count_trainable_params(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _apply_lora_if_enabled(config, model):
    if not config.use_lora:
        print("🔧 LoRA disabled: training full model parameters")
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "LoRA requested but package 'peft' is not installed. Run: pip install peft"
        ) from exc

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    print(
        "✅ LoRA enabled "
        f"(r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}, "
        f"targets={config.lora_target_modules})"
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def load_model_and_processor(config):
    """Load Whisper model + processor and optionally attach LoRA adapters."""
    print(f"🔧 Loading model: {config.model_name}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        config.model_name, language=config.language, task=config.task,
    )
    processor = WhisperProcessor.from_pretrained(
        config.model_name, language=config.language, task=config.task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)

    # Force Japanese generation config
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    model.generation_config.forced_decoder_ids = None

    model = _apply_lora_if_enabled(config, model)

    total_params = _count_total_params(model) / 1e6
    trainable_params = _count_trainable_params(model) / 1e6
    trainable_ratio = 100.0 * trainable_params / max(total_params, 1e-9)
    print(
        f"✅ Model ready: total={total_params:.1f}M, "
        f"trainable={trainable_params:.2f}M ({trainable_ratio:.2f}%)"
    )

    return model, processor, tokenizer, feature_extractor


def merge_lora_adapter_and_save(model, processor, output_dir):
    """Merge LoRA weights into base model and save full model for deployment."""
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "Cannot merge LoRA because package 'peft' is not installed."
        ) from exc

    if not isinstance(model, PeftModel):
        raise ValueError("merge_lora_adapter_and_save expects a PEFT model.")

    print(f"🔀 Merging LoRA adapter into base model -> {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("✅ Merged model saved")
    return merged_model


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad input features và labels, mask padding với -100."""
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Skip invalid samples
        features = [f for f in features if f.get("valid", True)]
        if not features:
            return None

        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def get_compute_metrics(tokenizer):
    """Return compute_metrics function using CER."""
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    return compute_metrics
