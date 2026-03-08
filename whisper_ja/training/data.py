"""
Dataset loading & preprocessing.
Cache xuống disk để training nhanh hơn.
"""

import os
import time

from datasets import load_dataset


def prepare_sample(batch, feature_extractor, tokenizer):
    """Chuyển audio → mel spectrogram, text → token IDs."""
    try:
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        with tokenizer.as_target_tokenizer():
            batch["labels"] = tokenizer(
                batch["transcription"],
                language="ja",
                task="transcribe",
            ).input_ids
        batch["valid"] = True
    except Exception:
        # File corrupt → tạo dummy data, sẽ filter sau
        batch["input_features"] = feature_extractor(
            [0.0] * 16000,  # 1 giây silence
            sampling_rate=16000,
        ).input_features[0]
        batch["labels"] = tokenizer("").input_ids
        batch["valid"] = False
    return batch


def load_and_prepare_train(config, feature_extractor, tokenizer):
    """Load ReazonSpeech + preprocess + cache to disk."""
    print(f"📥 Downloading ReazonSpeech '{config.reazonspeech_size}'...")
    start = time.perf_counter()

    kwargs = {"trust_remote_code": True}
    if config.hf_token:
        kwargs["token"] = config.hf_token

    dataset = load_dataset(
        "reazon-research/reazonspeech",
        config.reazonspeech_size,
        split="train",
        **kwargs,
    )
    print(f"✅ Train downloaded: {len(dataset):,} samples ({time.perf_counter() - start:.1f}s)")

    if config.max_train_samples and len(dataset) > config.max_train_samples:
        dataset = dataset.select(range(config.max_train_samples))
        print(f"✂️  Train subset enabled: {len(dataset):,} samples")

    num_proc = int(getattr(config, "num_proc", 1) or 1)
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") and num_proc > 1:
        # datasets+audio multiprocessing can hang on Kaggle runtime intermittently.
        print(f"⚠️  Kaggle runtime detected; forcing num_proc=1 (was {num_proc}) for stability")
        num_proc = 1

    print("⏳ Preprocessing train (cached to disk)...")
    pre_start = time.perf_counter()
    dataset = dataset.map(
        lambda batch: prepare_sample(batch, feature_extractor, tokenizer),
        remove_columns=dataset.column_names,
        num_proc=num_proc if num_proc > 1 else None,
        desc="preprocess-train",
    )

    dataset = dataset.filter(lambda x: x["valid"], desc="filter-train")
    print(f"✅ Train ready: {len(dataset):,} samples ({time.perf_counter() - pre_start:.1f}s)")

    return dataset


def load_and_prepare_eval(config, feature_extractor, tokenizer):
    """Load eval dataset + preprocess + cache to disk."""
    print(f"📥 Downloading eval dataset...")
    start = time.perf_counter()

    kwargs = {}
    if config.hf_token:
        kwargs["token"] = config.hf_token

    dataset = load_dataset(
        config.eval_dataset_name,
        split="test",
        **kwargs,
    )

    if config.max_eval_samples and len(dataset) > config.max_eval_samples:
        dataset = dataset.select(range(config.max_eval_samples))

    print(f"✅ Eval downloaded: {len(dataset):,} samples ({time.perf_counter() - start:.1f}s)")

    num_proc = int(getattr(config, "num_proc", 1) or 1)
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") and num_proc > 1:
        print(f"⚠️  Kaggle runtime detected; forcing num_proc=1 (was {num_proc}) for stability")
        num_proc = 1

    print("⏳ Preprocessing eval...")
    pre_start = time.perf_counter()
    dataset = dataset.map(
        lambda batch: prepare_sample(batch, feature_extractor, tokenizer),
        remove_columns=dataset.column_names,
        num_proc=num_proc if num_proc > 1 else None,
        desc="preprocess-eval",
    )

    print(f"✅ Eval ready: {len(dataset):,} samples ({time.perf_counter() - pre_start:.1f}s)")

    return dataset
