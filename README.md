# Whisper Tiny JA - LoRA Training (Private Repo)

Repo này là phần training private cho Japanese ASR. Mặc định dùng LoRA để tránh full finetune làm giảm chất lượng.

## Quick Start

```bash
# 1) Setup
bash setup.sh

# 2) Set token (bắt buộc nếu cần dataset gated hoặc push Hub)
export HF_TOKEN=hf_xxxxx

# 3) Train LoRA (default)
python train.py --reazonspeech_size small --batch_size 32 --num_train_epochs 3 --no_wandb
```

## LoRA Defaults

- LoRA được bật mặc định (`use_lora=True` trong `whisper_ja/config.py`).
- Sau khi train, script sẽ lưu:
  - Adapter: `output/whisper-tiny-ja-lora`
  - Merged full model (để deploy): `output/whisper-tiny-ja`
  - CT2 export: `output/whisper-tiny-ja-ct2`

CLI thường dùng:

```bash
# Tinh chỉnh LoRA
python train.py --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --lora_target_modules q_proj,v_proj

# Gắn tag cho W&B run
python train.py --wandb_tags kaggle,lora,reazonspeech-small

# Bật push Hub cho cả adapter + merged model
python train.py --push_to_hub \
  --hub_adapter_model_id dungca/whisper-tiny-ja-lora \
  --hub_model_id dungca/whisper-tiny-ja

# Nếu cần thử full finetune
python train.py --full_finetune
```

## Split Repo Architecture

- Repo này (`whisper-finetune-ja`): training private + CI trigger Kaggle.
- Repo demo riêng: Gradio/HF Space chỉ pull model từ Hub (merged model repo).

Chi tiết luồng triển khai xem: `deploy-project.md`.

## Module Layout

Code đã được tách module theo package `whisper_ja/`:

- `whisper_ja/config.py`: cấu hình train
- `whisper_ja/training/`: data/model/trainer/export/inference
- `whisper_ja/cli/`: entrypoint cho train, validation, quality gate, kaggle, hub tools

Các file root như `train.py`, `data_validation.py`, `kaggle_train.py` vẫn giữ để tương thích lệnh cũ.

## Kaggle Automation

Files đã có sẵn:
- `.github/workflows/ci.yml`
- `.github/workflows/trigger_kaggle_train.yml`
- `.github/workflows/promote_model.yml`
- `kernel-metadata.json`
- `kaggle_train.py`
- `.kaggleignore`
- `data_validation.py`
- `quality_gate.py`

Bạn cần:
1. Đổi `kernel-metadata.json` field `id` thành `your_kaggle_username/your-kernel-slug`.
2. Set GitHub Secrets trong repo private:
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`
   - `HF_TOKEN` (dùng cho data validation trong workflow CT)
   - `CROSS_REPO_PAT` (dùng để dispatch CD sang repo demo)
3. Set Kaggle Secret:
   - `HF_TOKEN`
   - `WANDB_API_KEY` (optional, để log training lên Weights & Biases)

Khi push lên `main`, workflow sẽ `kaggle kernels push -p .` để trigger training run trên Kaggle.
W&B được bật mặc định trong Kaggle run; nếu cần tắt thì set `ENABLE_WANDB=0`.
Bạn có thể gắn tag cho W&B run bằng env `WANDB_TAGS`, ví dụ: `WANDB_TAGS=kaggle,lora,reazonspeech-small`.

## MLOps Flow (CI + CT + CD)

### CI
- Workflow: `.github/workflows/ci.yml`
- Chạy khi push/PR.
- Kiểm tra:
  - Python syntax (`py_compile`)
  - `kernel-metadata.json` hợp lệ
  - Data validation lightweight trên eval dataset
- Report: artifact `ci-reports`

### CT (Continuous Training)
- Workflow: `.github/workflows/trigger_kaggle_train.yml`
- Trigger khi code train/data pipeline thay đổi hoặc chạy tay.
- Khi chạy tay (`workflow_dispatch`) có input `reazonspeech_size` (mặc định `small`).
- Trình tự:
  1. Validate metadata
  2. Validate training dataset quality (`data_validation.py`)
  3. Push kernel sang Kaggle để train LoRA
- Report: artifact `training-data-validation-report`

### CD (Promotion sang Demo)
- Workflow: `.github/workflows/promote_model.yml` (manual gate).
- Nhập `candidate_eval_cer`, `baseline_eval_cer`, `model_repo_id`, `demo_repo`.
- `quality_gate.py` quyết định `promote` hoặc `reject`.
- Nếu `promote`, workflow gửi `repository_dispatch` sang repo demo để update model.
- Report: artifact `quality-gate-report`

## Khi Dữ Liệu Update Thì Làm Gì

1. Commit thay đổi data config/pipeline vào repo training private.
2. CI chạy, kiểm tra syntax + validation lightweight.
3. CT chạy `data_validation.py` trên dataset train.
4. Nếu data pass, CT mới push train job lên Kaggle.
5. Sau khi có metric mới (CER), chạy `promote_model.yml` để quality gate.
6. Gate pass mới CD sang demo repo.

## Log Và Report Ở Đâu

- CI/CT/CD execution logs: tab **Actions** trên GitHub.
- Data quality reports: artifacts `ci-reports`, `training-data-validation-report`.
- Quality gate report: artifact `quality-gate-report`.
- Training curves/metrics: W&B (nếu bật) và logs trong Kaggle run.
