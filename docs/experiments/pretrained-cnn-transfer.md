# AudioSet-pretrained CNN transfer-learning study

更新日期：2026-08-17
狀態：development 選模與唯一 fold 10 最終評估完成

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
| Encoder learning rate | 0 for linear probe；1e-5 for partial fine-tuning v1；2e-5 for the only neighbouring candidate |
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
| 1 | 3 | 0.8796 | 0.8625 | 29.9 s | A100 |
| 4 | 5 | 0.8595 | 0.8646 | 24.0 s | A100 |
| 7 | 4 | 0.8023 | 0.8126 | 24.3 s | A100 |
| Mean ± std | - | 0.8471 ± 0.0327 | 0.8466 ± 0.0240 | - | - |

### Stage 2: partial fine-tuning

Linear probing 超越相同 development protocol 的 control `0.7818`，因此依預先定義規則進入 partial fine-tuning。第一個設定只解凍最後兩個 convolutional modules，encoder/head learning rates 分別為 `1e-5`/`3e-4`。

| Validation fold | Best epoch | Macro F1 | Accuracy | Time | GPU |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 3 | 0.8871 | 0.8660 | 41.5 s | A100 |
| 4 | 3 | 0.8900 | 0.8970 | 36.9 s | A100 |
| 7 | 5 | 0.8293 | 0.8496 | 36.2 s | A100 |
| Mean ± std | - | 0.8688 ± 0.0280 | 0.8709 ± 0.0196 | - | - |

Partial fine-tuning v1 比 linear probing 提高 `0.0217` Macro F1。依 protocol 最多再測一個鄰近設定；唯一改動是將 encoder learning rate 由 `1e-5` 提高至 `2e-5`，其餘條件固定。

| Validation fold | Best epoch | Macro F1 | Accuracy | Time | GPU |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 2 | 0.8882 | 0.8694 | 41.8 s | A100 |
| 4 | 5 | 0.8949 | 0.9010 | 36.3 s | A100 |
| 7 | 5 | 0.8317 | 0.8496 | 36.1 s | A100 |
| Mean ± std | - | 0.8716 ± 0.0283 | 0.8734 ± 0.0212 | - | - |

鄰近設定比 v1 提高 `0.0028` Macro F1，並比 from-scratch CNN control 高 `0.0898`。因此只依 development mean Macro F1 鎖定 encoder/head learning rates `2e-5`/`3e-4`，不再建立其他候選。

### Unique fold 10 final evaluation

鎖定設定後只執行一次 fold 10 test。Validation fold 4 在 development 階段具有最高 Macro F1，故在查看 test 前預先選為 final run 的 validation fold；test 結果沒有用於任何後續調參。

| Split | Accuracy | Macro F1 | Macro precision | Macro recall | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation fold 4 | 0.9010 | 0.8949 | - | - | 0.3145 |
| Sealed fold 10 test | 0.8949 | 0.9041 | 0.9179 | 0.8996 | 0.3537 |

Final run 的最佳 epoch 為 5。該 epoch 的 training Accuracy/Macro F1 為 `0.9482`/`0.9514`，相對 validation Macro F1 的差距約 `0.0565`，顯示仍有中度 generalisation gap，但沒有先前 from-scratch 訓練中接近飽和的嚴重過擬合。Test Macro F1 略高於 validation，且 test loss 接近 validation loss；因此沒有證據顯示 final checkpoint 在 fold 10 上崩潰。另一方面，fold 間 Macro F1 標準差仍約 `0.0283`，正式 10-fold cross-validation 仍是判斷穩健性的必要步驟。

EfficientAT 在 encoder 完全凍結時已達 `0.8471` mean Macro F1，表示 AudioSet representation 對 UrbanSound8K 有明顯可轉移性，而不是 underfitting。Partial fine-tuning 再帶來穩定但較小的改善。

### Artifacts

所有 checkpoints、histories、validation metrics、final test metrics 及 confusion matrix 已備份至：

```text
/content/drive/MyDrive/urbansound8k_data/experiment_artifacts/pretrained_cnn_transfer/
```

最終 run name 為 `pretrained_cnn_mn10_partial_ft_lr2e5_final_test_v1`。大型 artifacts 未提交 GitHub。

## 公平論文描述原則

- `0.7818` 是 from-scratch CNN 在相同三-fold development protocol 的 control mean，不是 fold 10 test。
- pretrained CNN 的優劣只以三個 development folds 的 mean/std 判斷。
- 不把 AudioSet 預訓練與 UrbanSound8K 從零訓練描述成相同資料條件；這個比較衡量的是 transfer learning 的價值。
- Pretrained model 超越 development control，但這不表示架構本身在相同訓練資料條件下優於 from-scratch CNN；改善包含大規模 AudioSet 預訓練的貢獻。
- Fold 10 只在唯一最終設定鎖定後評估一次，沒有用多個 test 候選挑最高分。
- 單一 fold 10 的 Macro F1 `0.9041` 是 final confirmation，不是 10-fold 泛化估計；正式結論應以後續固定設定的 10-fold mean/std 為準。
- 應同時報告預訓練成本、domain transfer、5 秒 zero-padding，以及三個 development folds 的變異。
