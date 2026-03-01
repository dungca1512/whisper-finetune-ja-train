#!/usr/bin/env python3
"""
=============================================================================
Whisper Tiny - Finetune for Japanese ASR
=============================================================================

Usage:
    # Train
    python train.py --reazonspeech_size small --batch_size 64 --num_proc 1

    # Resume từ checkpoint
    python train.py --reazonspeech_size small --batch_size 64 --num_proc 1 --resume

    # Export only
    python train.py --export_only

    # Test inference only
    python train.py --test_only
=============================================================================
"""

import os
import sys
import glob
import argparse
import traceback
from pathlib import Path

import torch
from huggingface_hub import HfApi

from dotenv import load_dotenv

from whisper_ja.config import Config
from whisper_ja.training.data import load_and_prepare_train, load_and_prepare_eval
from whisper_ja.training.model import (
    load_model_and_processor,
    merge_lora_adapter_and_save,
    DataCollatorSpeechSeq2SeqWithPadding,
    get_compute_metrics,
)
from whisper_ja.training.trainer import create_trainer
from whisper_ja.training.export import export_to_ct2
from whisper_ja.training.inference import test_inference

load_dotenv()


def setup_wandb(config):
    """Initialize W&B logging."""
    if not config.use_wandb:
        return

    import wandb

    key = config.wandb_key or os.getenv("WANDB_API_KEY", "")
    if key:
        wandb.login(key=key)
    else:
        print("⚠️  WANDB_API_KEY not set, skipping W&B")
        config.use_wandb = False
        return

    tags = list(config.wandb_tags)
    env_tags = os.getenv("WANDB_TAGS", "")
    if env_tags:
        tags.extend([tag.strip() for tag in env_tags.split(",") if tag.strip()])
    tags = list(dict.fromkeys(tags))

    wandb.init(
        project=config.wandb_project,
        name=f"reazonspeech-{config.reazonspeech_size}-{config.num_train_epochs}ep",
        tags=tags or None,
        config={
            "model": config.model_name,
            "dataset": f"reazonspeech-{config.reazonspeech_size}",
            "epochs": config.num_train_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        },
    )
    print("✅ W&B initialized")


def setup_hf_token(config):
    """Setup HuggingFace token."""
    token = config.hf_token or os.getenv("HF_TOKEN", "")
    if token:
        config.hf_token = token
        print("✅ HF token found")
    else:
        print("⚠️  HF_TOKEN not set — may fail on gated datasets")


def find_latest_checkpoint(output_dir):
    """Tìm checkpoint mới nhất trong output_dir."""
    checkpoints = sorted(glob.glob(f"{output_dir}/checkpoint-*"), key=lambda x: int(x.split("-")[-1]))
    if checkpoints:
        return checkpoints[-1]
    return None


def resolve_deploy_model_dir(config):
    """Model directory dùng cho deploy/export/inference."""
    if config.use_lora and config.save_merged_model and Path(config.merged_output_dir).is_dir():
        return config.merged_output_dir
    return config.output_dir


def push_folder_to_hub(folder_path, repo_id, token, private=False):
    """Upload local folder to Hugging Face Hub model repo."""
    if not token:
        print(f"⚠️  Skip push to {repo_id}: HF token is missing")
        return

    path = Path(folder_path)
    if not path.is_dir():
        print(f"⚠️  Skip push to {repo_id}: folder not found ({folder_path})")
        return

    print(f"🌐 Uploading {folder_path} -> https://huggingface.co/{repo_id}")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(path),
        ignore_patterns=["checkpoint-*", "logs/*", "optimizer.pt", "scheduler.pt", "rng_state.pth"],
        commit_message="Update model artifacts from automated training run",
    )
    print("✅ Upload complete")


def train(config, args):
    """Full training pipeline."""
    print("=" * 60)
    print("🎌 Whisper Tiny — Japanese ASR LoRA Training")
    print(f"   Mode: {'LoRA adapters' if config.use_lora else 'Full finetune'}")
    print("=" * 60)

    # Setup
    setup_hf_token(config)
    setup_wandb(config)

    # Load model
    model, processor, tokenizer, feature_extractor = load_model_and_processor(config)

    # Load & preprocess data (cached to disk)
    train_dataset = load_and_prepare_train(config, feature_extractor, tokenizer)
    eval_dataset = load_and_prepare_eval(config, feature_extractor, tokenizer)

    print(f"📊 Train: {len(train_dataset):,} samples")
    print(f"📊 Eval:  {len(eval_dataset):,} samples")

    # Collator & metrics
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    compute_metrics = get_compute_metrics(tokenizer)

    # Create trainer
    trainer, device = create_trainer(
        config, model, processor, tokenizer,
        train_dataset, eval_dataset,
        data_collator, compute_metrics,
    )

    # Resume from checkpoint?
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = find_latest_checkpoint(config.output_dir)
        if resume_checkpoint:
            print(f"🔄 Resuming from {resume_checkpoint}")
        else:
            print("⚠️  No checkpoint found, training from scratch")

    # Train
    print(f"\n🚀 Starting training...")
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Epochs: {config.num_train_epochs}")
    print(f"   Steps/epoch: ~{len(train_dataset) // config.batch_size:,}")
    print()

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    print("\n✅ Training complete!")

    # Save
    print(f"\n💾 Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)

    deploy_model_dir = config.output_dir

    # Evaluate
    print("\n📊 Final evaluation...")
    eval_metrics = trainer.evaluate()

    print(f"\n{'='*40}")
    print(f"📊 Final CER:  {eval_metrics['eval_cer']:.4f} ({eval_metrics['eval_cer']*100:.2f}%)")
    print(f"📊 Eval Loss:  {eval_metrics['eval_loss']:.4f}")
    print(f"{'='*40}")

    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", eval_metrics)

    # W&B cleanup
    if config.use_wandb:
        import wandb
        wandb.finish()

    # Merge LoRA adapter for deployment/export after metrics are finalized.
    if config.use_lora and config.save_merged_model:
        model_to_merge = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        merge_lora_adapter_and_save(
            model_to_merge,
            processor,
            output_dir=config.merged_output_dir,
        )
        deploy_model_dir = config.merged_output_dir

    # Push to hub
    if config.push_to_hub:
        if config.use_lora:
            push_folder_to_hub(
                folder_path=config.output_dir,
                repo_id=config.hub_adapter_model_id,
                token=config.hf_token,
            )
            if config.save_merged_model and config.push_merged_to_hub:
                push_folder_to_hub(
                    folder_path=deploy_model_dir,
                    repo_id=config.hub_model_id,
                    token=config.hf_token,
                )
        else:
            push_folder_to_hub(
                folder_path=config.output_dir,
                repo_id=config.hub_model_id,
                token=config.hf_token,
            )

    # Export
    export_to_ct2(config, model_dir=deploy_model_dir)

    # Test
    if config.run_post_train_test:
        print("\n" + "=" * 60)
        try:
            test_inference(config, device=device, num_samples=8, model_dir=deploy_model_dir)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️  Post-train inference test failed: {exc}")
            traceback.print_exc()
    else:
        print("\n⏭️  Skipping post-train inference test")

    print("\n🎉 All done!")
    print(f"   Adapter: {config.output_dir}")
    print(f"   Deploy:  {deploy_model_dir}")
    if config.export_ct2:
        print(f"   CT2:   {config.ct2_output_dir}")


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Finetune Whisper Tiny for Japanese ASR")

    # Actions
    parser.add_argument("--export_only", action="store_true", help="Export existing model to CT2")
    parser.add_argument("--test_only", action="store_true", help="Run inference test only")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--skip_final_test", action="store_true", help="Skip post-train inference comparison test")

    # Dataset
    parser.add_argument("--reazonspeech_size", type=str, help="tiny/small/medium/large/all")
    parser.add_argument("--max_eval_samples", type=int, help="Limit eval samples")

    # Training
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--output_dir", type=str, help="Adapter output directory")
    parser.add_argument("--merged_output_dir", type=str, help="Merged model output directory")
    parser.add_argument("--full_finetune", action="store_true", help="Disable LoRA and train all params")
    parser.add_argument("--no_merge_lora", action="store_true", help="Do not save merged full model after LoRA training")
    parser.add_argument("--lora_r", type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, help="Comma-separated target modules, e.g. q_proj,v_proj")

    # Tokens
    parser.add_argument("--hf_token", type=str, help="HuggingFace token")
    parser.add_argument("--wandb_key", type=str, help="W&B API key")
    parser.add_argument("--wandb_tags", type=str, help="Comma-separated W&B tags")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B")
    parser.add_argument("--num_proc", type=int, help="CPU cores for preprocessing")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload artifacts to HF Hub")
    parser.add_argument("--hub_model_id", type=str, help="HF repo id for deploy (merged/full) model")
    parser.add_argument("--hub_adapter_model_id", type=str, help="HF repo id for LoRA adapter model")
    parser.add_argument("--adapter_only_hub", action="store_true", help="Only push LoRA adapter, skip merged model push")

    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    # Override config with CLI args
    for key, value in vars(args).items():
        if key in (
            "export_only",
            "test_only",
            "no_wandb",
            "resume",
            "full_finetune",
            "no_merge_lora",
            "lora_target_modules",
            "wandb_tags",
            "skip_final_test",
            "push_to_hub",
            "adapter_only_hub",
        ):
            continue
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    if args.no_wandb:
        config.use_wandb = False
    if args.full_finetune:
        config.use_lora = False
    if args.no_merge_lora:
        config.save_merged_model = False
    if args.push_to_hub:
        config.push_to_hub = True
    if args.adapter_only_hub:
        config.push_merged_to_hub = False
    if args.lora_target_modules:
        config.lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    if args.wandb_tags:
        config.wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    if args.skip_final_test:
        config.run_post_train_test = False

    # Actions
    if args.export_only:
        export_to_ct2(config, model_dir=resolve_deploy_model_dir(config))
    elif args.test_only:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        test_inference(config, device=device, model_dir=resolve_deploy_model_dir(config))
    else:
        train(config, args)


if __name__ == "__main__":
    main()
