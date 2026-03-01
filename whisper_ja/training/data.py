"""
Dataset loading & preprocessing.
Cache xuống disk để training nhanh hơn.
"""

from datasets import load_dataset


def prepare_sample(batch, feature_extractor, tokenizer):
    """Chuyển audio → mel spectrogram, text → token IDs."""
    try:
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
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

    kwargs = {"trust_remote_code": True}
    if config.hf_token:
        kwargs["token"] = config.hf_token

    dataset = load_dataset(
        "reazon-research/reazonspeech",
        config.reazonspeech_size,
        split="train",
        **kwargs,
    )
    print(f"✅ Train downloaded: {len(dataset):,} samples")

    print("⏳ Preprocessing train (cached to disk)...")
    dataset = dataset.map(
        lambda batch: prepare_sample(batch, feature_extractor, tokenizer),
        remove_columns=dataset.column_names,
        num_proc=config.num_proc,
    )

    print(f"✅ Train ready: {len(dataset):,} samples")

    return dataset


def load_and_prepare_eval(config, feature_extractor, tokenizer):
    """Load eval dataset + preprocess + cache to disk."""
    print(f"📥 Downloading eval dataset...")

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

    print(f"✅ Eval downloaded: {len(dataset):,} samples")

    print("⏳ Preprocessing eval...")
    dataset = dataset.map(
        lambda batch: prepare_sample(batch, feature_extractor, tokenizer),
        remove_columns=dataset.column_names,
        num_proc=config.num_proc,
    )

    print(f"✅ Train ready: {len(dataset):,} samples")

    return dataset
