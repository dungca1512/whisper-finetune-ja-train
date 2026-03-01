# Deployment Blueprint: Split Repo (Training Private + Demo Repo)

## 1) Mục tiêu

Bạn đang dùng 2 repo riêng:

- **Repo A (private):** training LoRA + automation (repo hiện tại)
- **Repo B (public/private):** demo app (Gradio / Hugging Face Space)

Luồng chuẩn:

1. Repo A train LoRA trên Kaggle
2. Repo A merge adapter -> full model
3. Repo A push model merged lên Hugging Face Hub
4. Repo B / HF Space load model merged mới nhất từ Hub

## 2) Repo A (Training Private)

### 2.1 File bắt buộc

- `train.py`: train LoRA mặc định
- `whisper_ja/config.py`: cấu hình LoRA + output
- `data_validation.py`: kiểm tra chất lượng dữ liệu (gate trước CT)
- `quality_gate.py`: quyết định promote/reject model
- `kaggle_train.py`: entrypoint chạy trên Kaggle
- `kernel-metadata.json`: cấu hình Kaggle kernel
- `.github/workflows/ci.yml`: CI checks
- `.github/workflows/trigger_kaggle_train.yml`: trigger `kaggle kernels push`
- `.github/workflows/promote_model.yml`: quality gate + dispatch CD sang demo
- `.kaggleignore`: loại trừ file local khi push kernel

### 2.2 Secrets cần có

Trong GitHub repo private:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- `HF_TOKEN` (dùng trong `data_validation.py` khi dataset cần auth)
- `CROSS_REPO_PAT` (dispatch CD sang repo demo)

Trong Kaggle Secrets:

- `HF_TOKEN`

### 2.3 Cấu hình quan trọng

`kernel-metadata.json` phải đổi `id` theo tài khoản thật:

```json
{
  "id": "your_kaggle_username/whisper-ja-lora-train"
}
```

### 2.4 Trigger train

- Push vào nhánh `main` (các file train) -> GitHub Action gọi `kaggle kernels push -p .`
- Hoặc chạy thủ công `workflow_dispatch` với input `reazonspeech_size`
- Trước khi push Kaggle, workflow chạy data validation và lưu JSON report artifact.

## 3) Repo B (Demo Repo)

Repo demo chỉ cần inference app và dependencies runtime.

Cấu trúc gợi ý:

```text
asr-ja-demo/
├── app.py
├── requirements.txt
├── packages.txt
└── README.md
```

`app.py` load model merged từ Hub, ví dụ:

```python
from transformers import pipeline

MODEL_ID = "dungca/whisper-tiny-ja"
asr = pipeline("automatic-speech-recognition", model=MODEL_ID)
```

## 4) Naming model repos trên HF Hub

Khuyến nghị tách 2 model repo trên HF:

- `dungca/whisper-tiny-ja-lora` (adapter)
- `dungca/whisper-tiny-ja` (merged full model cho demo)

Demo repo luôn dùng repo merged để tránh phụ thuộc logic load adapter.

## 5) Checklist rollout

1. Đặt repo này là private trên GitHub.
2. Cập nhật `kernel-metadata.json` (đúng username Kaggle).
3. Set GitHub Secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `HF_TOKEN`, `CROSS_REPO_PAT`.
4. Set Kaggle Secret: `HF_TOKEN`.
5. Push PR/commit để CI chạy (`ci.yml`).
6. Push `main` để trigger CT (`trigger_kaggle_train.yml`).
7. Kiểm tra Kaggle kernel status + artifacts validation.
8. Lấy CER candidate và baseline, chạy `promote_model.yml`.
9. Nếu quality gate pass -> workflow dispatch CD sang repo demo.
10. Demo repo trỏ tới `hub_model_id` merged.

## 6) Quy trình CI / CT / CD Khi Data Update

1. Data hoặc data pipeline thay đổi trong repo training.
2. CI chạy syntax + metadata checks + lightweight dataset validation.
3. CT chạy full hơn trên data train sample:
   - `data_validation.py` tạo report
   - Nếu report pass -> push Kaggle train
4. Sau train xong, lấy CER candidate.
5. Chạy `promote_model.yml` với candidate CER và baseline CER.
6. `quality_gate.py` quyết định:
   - `promote`: dispatch sang demo repo để update model.
   - `reject`: giữ model ở trạng thái candidate, không deploy.

## 7) Log Và Report

- **GitHub Actions logs:** toàn bộ CI/CT/CD run logs.
- **Artifacts:**
  - `ci-reports` (CI validation)
  - `training-data-validation-report` (CT data quality)
  - `quality-gate-report` (CD decision)
- **Kaggle run logs:** chi tiết train runtime.
- **W&B (nếu bật):** tracking loss/CER theo run.

## 8) Lưu ý vận hành

- LoRA là mặc định để giảm rủi ro quality regression so với full finetune.
- Nếu cần A/B, bật `--full_finetune` cho run riêng và so CER trước khi dùng cho production/demo.
- Không dùng `--force` push git giữa 2 repo; chỉ đồng bộ thông qua model artifact trên Hub.
