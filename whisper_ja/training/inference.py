"""
Test inference - so sánh original vs finetuned model.
"""

import json
import os

from datasets import load_dataset
from transformers import pipeline


def _is_adapter_only_dir(model_dir):
    if not os.path.isdir(model_dir):
        return False
    adapter_cfg = os.path.join(model_dir, "adapter_config.json")
    has_adapter = os.path.isfile(adapter_cfg)
    has_full_weights = (
        os.path.isfile(os.path.join(model_dir, "model.safetensors")) or
        os.path.isfile(os.path.join(model_dir, "pytorch_model.bin"))
    )
    return has_adapter and not has_full_weights


def _build_finetuned_pipeline(model_ref, device):
    if not _is_adapter_only_dir(model_ref):
        return pipeline(
            "automatic-speech-recognition",
            model=model_ref,
            device=device,
        )

    try:
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Adapter model detected but missing dependencies (peft/transformers)."
        ) from exc

    with open(os.path.join(model_ref, "adapter_config.json"), "r", encoding="utf-8") as fh:
        adapter_config = json.load(fh)
    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise RuntimeError("Cannot load adapter: adapter_config.json missing base_model_name_or_path.")

    print(f"🔌 Adapter-only model detected. Loading base model: {base_model_name}")
    processor = WhisperProcessor.from_pretrained(model_ref)
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, model_ref)
    merged_model = peft_model.merge_and_unload()

    return pipeline(
        "automatic-speech-recognition",
        model=merged_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
    )


def test_inference(config, device="cuda:0", num_samples=10, model_dir=None):
    """So sánh kết quả Original vs Finetuned."""
    print("🎤 Loading models for comparison...")
    finetuned_model_ref = model_dir or config.output_dir

    pipe_finetuned = _build_finetuned_pipeline(finetuned_model_ref, device=device)
    pipe_original = pipeline(
        "automatic-speech-recognition",
        model=config.model_name,
        device=device,
    )

    kwargs = {}
    if config.hf_token:
        kwargs["token"] = config.hf_token

    test_data = load_dataset(
        config.eval_dataset_name,
        split="test",
        **kwargs,
    )

    gen_kwargs = {
        "language": config.language,
        "task": config.task,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
        "max_new_tokens": 128,
    }

    print(f"\n{'='*60}")
    print(f"Original vs Finetuned ({num_samples} samples)")
    print(f"{'='*60}\n")

    for i in range(min(num_samples, len(test_data))):
        sample = test_data[i]
        audio = sample["audio"]["array"]

        r_orig = pipe_original(audio, generate_kwargs=gen_kwargs)
        r_ft = pipe_finetuned(audio, generate_kwargs=gen_kwargs)

        print(f"Sample {i+1}:")
        print(f"  📝 Reference:  {sample['transcription']}")
        print(f"  ❌ Original:   {r_orig['text']}")
        print(f"  ✅ Finetuned:  {r_ft['text']}")
        print()
