# Whisper JA - LoRA Training

Repo này là phần training private cho Japanese ASR. Mặc định dùng LoRA để tránh full finetune làm giảm chất lượng.

## Quick Start

```bash
# 1) Setup
bash setup.sh

# 2) Tạo env local
cp .env.example .env
# chỉ điền secrets trong .env (HF_TOKEN, WANDB_API_KEY, ...)

# 3) Train LoRA (default)
python train.py --reazonspeech_size small --batch_size 32 --num_train_epochs 3 --no_wandb
```

## Latest Training Metrics (W&B)

Nguồn: W&B `Run summary` từ Kaggle run (GPU P100, LoRA).

- `eval/cer`: **0.52497** (~52.50%)
- `eval/loss`: **1.17656**
- `eval/runtime`: **162.422s**
- `eval/samples_per_second`: **12.314**
- `eval/steps_per_second`: **0.77**
- `train/global_step`: **3000**
- `train/epoch`: **1.54719**
- `train/grad_norm`: **2.16062**
- `train/learning_rate`: **1e-5**

Gợi ý ghi model card trên HF:
- `CER`: `0.52497`
- `Eval loss`: `1.17656`

## LoRA Defaults

- LoRA được bật mặc định (`use_lora=True` trong `whisper_ja/config.py`).
- Đổi size model trong `whisper_ja/config.py` hoặc override tạm bằng CLI `--model_size`.
- Sau khi train, script sẽ lưu:
  - Adapter: `output/whisper-<model_size>-ja-lora`
  - Merged full model (để deploy): `output/whisper-<model_size>-ja`
  - CT2 export: `output/whisper-<model_size>-ja-ct2`

### Config Policy

- `.env`:
  - Chỉ giữ token/secret: `HF_TOKEN`, `WANDB_API_KEY`, `KAGGLE_KEY`, `GITHUB_TOKEN`.
- `whisper_ja/config.py`:
  - Giữ toàn bộ non-secret: model/repo identity, siêu tham số train/eval/data, output/merged/ct2 path.

CLI thường dùng:

```bash
# Tinh chỉnh LoRA
python train.py --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --lora_target_modules q_proj,v_proj

# Gắn tag cho W&B run
python train.py --wandb_tags kaggle,lora,reazonspeech-small

# Bật push Hub cho cả adapter + merged model
python train.py --push_to_hub \
  --hub_adapter_model_id <repo_owner>/whisper-<model_size>-ja-lora \
  --hub_model_id <repo_owner>/whisper-<model_size>-ja

# Nếu cần thử full finetune
python train.py --full_finetune

# Bỏ test inference cuối run (hữu ích cho môi trường CI/Kaggle)
python train.py --skip_final_test
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
3. Không bắt buộc set secret thủ công trong Kaggle Notebook.
   Workflow CT sẽ tự tạo `runtime_secrets.json` từ GitHub Secrets và đẩy kèm kernel run.
4. Nếu muốn fallback bằng Kaggle UI (tuỳ chọn), bạn vẫn có thể thêm:
   - `HF_TOKEN`
   - `WANDB_API_KEY`
   theo đường dẫn `Edit Notebook` -> `Add-ons` -> `Secrets`.

Khi push lên `main`, workflow sẽ `kaggle kernels push -p .` để trigger training run trên Kaggle.
W&B bật/tắt và tags được quản lý trong `whisper_ja/config.py` (`use_wandb`, `wandb_tags`).
Đổi size model cho Kaggle bằng `model_size` trong `whisper_ja/config.py` (hoặc override tạm qua CLI).
Bật/tắt post-train inference test bằng `run_post_train_test` trong `whisper_ja/config.py`.

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
