# AudioSet-pretrained CNN transfer-learning study

更新日期：2026-08-17
狀態：實作完成，development linear probing 待在 Colab A100 執行

## 研究目的

本實驗加入第三個模型 `EfficientAT mn10_as`，用於比較：

1. from-scratch CNN（主要基準）；
2. from-scratch Spectrogram Transformer（架構比較）；
3. AudioSet-pretrained CNN（transfer-learning 比較）。

選模的唯一主要指標是 folds 1、4、7 的平均 validation Macro F1。Accuracy 僅作輔助指標。Fold 10 在唯一設定鎖定前完全封存，禁止用 test 指標調整模型或超參數。

## 模型選擇

- 模型：EfficientAT `mn10_as`
- 架構：MobileNetV3-style CNN，包含 inverted residual blocks 與 squeeze-and-excitation
- 上游版本：commit `a425fdce92572e602a1d5634799bd9f1f2efa806`
- 預訓練資料：AudioSet；上游說明亦指出預設先做 ImageNet initialization，再訓練 AudioSet
- AudioSet checkpoint：`mn10_as_mAP_471.pt`
- 官方規模：4.88M parameters、0.54 GMACs（10 秒輸入）
- 官方 AudioSet performance：mAP 0.471
- 授權：MIT License；授權全文保存在 `third_party/efficientat/LICENSE`

選用 `mn10_as` 而非 PANNs CNN14 或大型 Audio Spectrogram Transformer 的理由，是其運算量及參數量明顯較小，能在 Colab GPU 上快速做三個 development folds，同時保留 AudioSet transfer learning 的研究價值。

## 官方 waveform preprocessing

本實驗不把既有、逐 clip 標準化的 Mel cache 插值後送入模型。輸入從原始 waveform 開始，遵循 EfficientAT 官方 downstream 範例：

- sampling rate：32 kHz；
- waveform：mono；
- waveform length：5 秒，短音訊在右側補零，長音訊從開頭截斷；UrbanSound8K clips 最長約 4 秒，因此不會截掉原事件；
- pre-emphasis coefficient：0.97；
- STFT：1024-point FFT、800-sample Hann window、320-sample hop；
- Mel resolution：128 bins；
- log compression：`log(mel + 1e-5)`；
- normalization：`(log_mel + 4.5) / 5`；
- linear probing 首輪關閉 frequency/time masking 及頻率邊界 augmentation，以隔離 pretrained representation 的效果。

為避免每個 epoch 重複解碼及 resample，允許建立獨立的 waveform-only cache：

```text
data/processed/urbansound8k_waveforms_32k_5s/
```

此 cache 只保存 32 kHz、5 秒 waveform；官方 log-Mel feature extraction 仍在 training batch 載入後於 GPU 動態執行。原始音訊及既有 Mel cache不會被刪除或覆寫。

## Development protocol

| 項目 | 固定值 |
| --- | --- |
| Sealed test fold | 10 |
| Development validation folds | 1, 4, 7 |
| Seed | 42 |
| Selection metric | Mean validation Macro F1 |
| Auxiliary metric | Mean validation Accuracy |
| Linear-probe epochs | 5 |
| Optimizer | AdamW |
| Head learning rate | 3e-4 |
| Encoder learning rate | 0 for linear probe; 1e-5 for partial fine-tuning |
| Weight decay | 1e-4 |
| Class weighting / sampling | power 0.5 / power 0.5 |

Linear probing 凍結全部 encoder 及 pretrained MLP representation layers，只訓練新的 UrbanSound8K 10-class final linear layer。每個 epoch 都保存 training/validation metrics，並只依 validation Macro F1 保存最佳 checkpoint。

若三-fold 平均 Macro F1 接近 control `0.7818`，或五個 epochs 有清楚上升趨勢，第二階段從每個 fold 的最佳 linear-probe checkpoint 接續，只解凍 encoder 最後兩個 convolutional modules，並使用 encoder/head differential learning rates。若仍明顯沒有競爭力，停止實驗，不用增加 epochs 掩蓋結果。

## 執行命令

建立獨立 waveform cache：

```text
python3 scripts/cache_urbansound8k_waveforms.py \
  --raw-dir data/raw/UrbanSound8K_soundata \
  --out-dir data/processed/urbansound8k_waveforms_32k_5s
```

執行 linear probing：

```text
python3 scripts/run_pretrained_cnn_transfer.py \
  --config configs/pretrained_cnn_linear_probe.yaml \
  --backup-root /content/drive/MyDrive/urbansound8k_data/experiment_artifacts/pretrained_cnn_transfer
```

只有 linear probing 達到預定競爭力門檻時，才執行：

```text
python3 scripts/run_pretrained_cnn_transfer.py \
  --config configs/pretrained_cnn_partial_finetune.yaml \
  --backup-root /content/drive/MyDrive/urbansound8k_data/experiment_artifacts/pretrained_cnn_transfer
```

## 實驗輸出

每個 validation fold 保存：

- `history.csv`
- `validation_metrics.json`
- `experiment_manifest.json`
- `config_resolved.json`
- `best_model.pt`
- `training_history.png`

三-fold 完成後保存：

- `development_summary.json`
- `development_summary.csv`

大型 cache、checkpoints、results 及 figures 不提交 GitHub；每個完成 run 立即備份到 Google Drive。

## 結果表

### Stage 1: linear probing

| Validation fold | Best epoch | Macro F1 | Accuracy | Time | GPU |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 待執行 | 待執行 | 待執行 | 待執行 | A100 preferred |
| 4 | 待執行 | 待執行 | 待執行 | 待執行 | A100 preferred |
| 7 | 待執行 | 待執行 | 待執行 | 待執行 | A100 preferred |
| Mean ± std | - | 待執行 | 待執行 | - | - |

### Stage 2: partial fine-tuning

尚未觸發；必須先依 Stage 1 的 development validation Macro F1 決定。

## 公平論文描述原則

- `0.7818` 是 from-scratch CNN 在相同三-fold development protocol 的 control mean，不是 fold 10 test。
- pretrained CNN 的優劣只以三個 development folds 的 mean/std 判斷。
- 不把 AudioSet 預訓練與 UrbanSound8K 從零訓練描述成相同資料條件；這個比較衡量的是 transfer learning 的價值。
- 若 pretrained model 未超越 CNN，仍應報告其較小運算量、收斂速度、可能的 domain mismatch 與短音訊 zero-padding 限制。
- Fold 10 只能在唯一最終設定鎖定後評估一次，不能用多個 test 候選挑最高分。
