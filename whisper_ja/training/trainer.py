"""
Training logic.
"""

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)


def detect_hardware():
    """Detect GPU and precision support."""
    if not torch.cuda.is_available():
        print("⚠️  No GPU detected, training on CPU")
        return "cpu", False, False

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    # BF16 support: Ampere (RTX 30xx) trở lên
    use_bf16 = any(k in gpu_name for k in ["RTX 30", "RTX 40", "RTX 50", "A100", "H100", "L40"])
    use_fp16 = not use_bf16

    print(f"🖥️  GPU: {gpu_name} ({vram:.1f}GB VRAM)")
    print(f"📐 Precision: {'bf16' if use_bf16 else 'fp16'}")

    return "cuda:0", use_fp16, use_bf16


def create_trainer(
    config, model, processor, tokenizer,
    train_dataset, eval_dataset,
    data_collator, compute_metrics,
):
    """Create Seq2SeqTrainer with all settings."""
    device, use_fp16, use_bf16 = detect_hardware()

    # W&B or tensorboard
    report_to = ["wandb"] if config.use_wandb else ["tensorboard"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,

        # Training
        gradient_checkpointing=config.gradient_checkpointing,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        fp16=use_fp16,
        bf16=use_bf16,

        # Evaluation
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        predict_with_generate=True,
        generation_max_length=225,

        # Saving
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,

        # Logging
        logging_steps=config.logging_steps,
        logging_dir=f"{config.output_dir}/logs",
        report_to=report_to,

        # Hub push handled manually after training (adapter + merged model)
        push_to_hub=False,
        hub_model_id=None,

        # Performance
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience
        )],
    )

    return trainer, device
