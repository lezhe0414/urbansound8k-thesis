# EfficientAT pretrained CNN v2 development study

更新日期：2026-08-17
狀態：已完成；只使用 development validation，未執行 v2 fold 10 test

## 研究目的與 protocol

本研究延伸已完成一次封存 test evaluation 的 EfficientAT v1。V2 只使用 UrbanSound8K folds 1、4、7 作為 development validation folds，並以三個 folds 的平均 validation Macro F1 作為唯一主要選模指標。Validation Accuracy 只作輔助。V1 的 fold 10 Macro F1 `0.9041` 不參與任何 v2 決策，v2 也沒有再次執行 fold 10 test。

所有候選均固定 EfficientAT `mn10_as` AudioSet-pretrained checkpoint、seed 42、官方 32 kHz waveform frontend、5 秒輸入、class weighting、class-aware sampling、AdamW、head learning rate `3e-4` 及 encoder learning rate `2e-5`。每個 fold 只依 validation Macro F1 保存最佳 checkpoint。

V1 development reference 為 Macro F1 `0.8716 ± 0.0283`、Accuracy `0.8734 ± 0.0212`。From-scratch CNN 的相同三-fold control Macro F1 為 `0.7818`。

## 第一階段：受控 A-D 比較

Stage A 先把 epochs 由 5 增加至 8，其餘條件不變。Stages B-D 各自只相對 Stage A 改動一類變因。

| 階段 | Run name | 唯一變因 | Macro F1 mean ± std | Accuracy mean ± std | 耗時 | 決策 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| V1 | `pretrained_cnn_mn10_partial_ft_lr2e5_v1` | 5 epochs、last 2 blocks | 0.8716 ± 0.0283 | 0.8734 ± 0.0212 | - | 歷史參考 |
| A | `pretrained_cnn_mn10_v2_epochs8` | epochs 5 → 8 | 0.8774 ± 0.0265 | 0.8777 ± 0.0222 | 178.6 s | 暫時保留 |
| B | `pretrained_cnn_mn10_v2_gradual8_head2` | 前 2 epochs head-only，再解凍 last 2 | 0.8772 ± 0.0265 | 0.8771 ± 0.0210 | 174.0 s | 未超越 A |
| C | `pretrained_cnn_mn10_v2_mask_f8_t24` | official frontend frequency/time masks 8/24 | 0.8771 ± 0.0300 | 0.8762 ± 0.0278 | 186.4 s | 未超越 A，變異較大 |
| D | `pretrained_cnn_mn10_v2_mixup_a015_p050` | post-frontend Mixup alpha 0.15、probability 0.5 | 0.8786 ± 0.0203 | 0.8788 ± 0.0174 | 188.3 s | A-D 勝出 |

### Per-fold results

| 階段 | Fold 1 F1 / best epoch | Fold 4 F1 / best epoch | Fold 7 F1 / best epoch |
| --- | ---: | ---: | ---: |
| A | 0.8882 / 2 | 0.9030 / 8 | 0.8409 / 6 |
| B | 0.8920 / 3 | 0.8996 / 8 | 0.8400 / 6 |
| C | 0.8855 / 8 | 0.9089 / 8 | 0.8369 / 8 |
| D | 0.8892 / 2 | 0.8965 / 6 | 0.8502 / 6 |

延長至 8 epochs 相較 v1 提高平均 Macro F1 約 `0.0058`。Gradual unfreezing 沒有穩健超越直接 partial fine-tuning。輕量 masking 的三個最佳 epoch 都落在 epoch 8，但平均值沒有提升且 fold 間變異增加。Weak Mixup 的平均值最高，並把標準差由 Stage A 的 `0.0265` 降至 `0.0203`，因此只以 D 為基礎進入解凍深度比較。

## 第二階段：解凍深度

此階段固定 Stage D 的所有條件，只比較最後 1、2、3 個 convolution blocks。Last 2 直接使用 Stage D 結果，僅新增 last 1 與 last 3 runs。

| 解凍深度 | Run name | Macro F1 mean ± std | Accuracy mean ± std | 耗時 | 決策 |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `pretrained_cnn_mn10_v2_mixup_a015_p050_last1` | 0.8706 ± 0.0354 | 0.8719 ± 0.0241 | 172.9 s | 未保留；fold 7 明顯下降 |
| 2 | `pretrained_cnn_mn10_v2_mixup_a015_p050` | 0.8786 ± 0.0203 | 0.8788 ± 0.0174 | 188.3 s | 中間點 |
| 3 | `pretrained_cnn_mn10_v2_mixup_a015_p050_last3` | **0.8844 ± 0.0165** | **0.8824 ± 0.0141** | 200.4 s | **唯一勝出設定** |

| 解凍深度 | Fold 1 F1 / best epoch | Fold 4 F1 / best epoch | Fold 7 F1 / best epoch |
| ---: | ---: | ---: | ---: |
| 1 | 0.8981 / 3 | 0.8929 / 6 | 0.8206 / 7 |
| 2 | 0.8892 / 2 | 0.8965 / 6 | 0.8502 / 6 |
| 3 | 0.8957 / 3 | 0.8963 / 7 | 0.8610 / 8 |

Last 3 相較 last 2 提高 Macro F1 `0.0057`，相較 v1 提高 `0.0128`，且標準差更低。主要改善來自 fold 7，而 folds 1、4 沒有犧牲。這個受控結果不支持在目前深度下發生明顯 catastrophic forgetting；但沒有測試解凍更多 blocks，因此不能推論 full fine-tuning 也會改善。

## 唯一最佳設定

設定檔：`configs/pretrained_cnn_v2_mixup_last3.yaml`

| 項目 | 數值 |
| --- | --- |
| Model | EfficientAT MN10 |
| Pretrained data | AudioSet |
| Total parameters | 4,214,554 |
| Trainable / frozen parameters | 1,763,050 / 2,451,504 |
| Trainable encoder depth | Last 3 convolution blocks |
| Optimizer | AdamW |
| Encoder / head LR | `2e-5` / `3e-4` |
| Epochs / seed | 8 / 42 |
| Mixup | alpha 0.15, probability 0.5 |
| Validation selection | Per-fold best Macro F1 |
| Input | 32 kHz mono waveform, 5 seconds |
| Official frontend | EfficientAT `AugmentMelSTFT` |
| Frontend details | FFT 1024, win 800, hop 320, 128 mel bins |
| Log-Mel normalization | `(log_mel + 4.5) / 5` |
| Mean validation Macro F1 | **0.8844 ± 0.0165** |
| Mean validation Accuracy | **0.8824 ± 0.0141** |

相較 from-scratch three-fold control `0.7818`，唯一最佳設定的 Macro F1 高 `0.1026`。此差異同時包含 AudioSet pretraining、不同官方 frontend 與 transfer-learning protocol 的影響，不能描述成純 CNN 架構的公平增益。

## 收斂與過擬合判讀

- Fold 1 在 epoch 3 達到最佳 Macro F1 `0.8957`，後續有輕微 validation 退化，顯示較早收斂。
- Fold 4 在 epoch 7 達到 `0.8963`，epoch 8 幾乎持平。
- Fold 7 在 epoch 8 達到 `0.8610`，仍呈改善，支持 8 epochs 對較難 fold 有價值。
- 三個 folds 的最佳 epoch 不一致，因此使用 per-fold validation Macro F1 checkpoint selection 比固定取最後 epoch 更合理。
- Mixup 與 class-aware sampling 會刻意降低 training Accuracy/F1；training metrics 是相對原始標籤的 proxy，不能直接與乾淨 validation metrics 比較，也不能據此判定 underfitting。
- 整體沒有單一「嚴重 overfitting」訊號；fold 1 有早期 peak，fold 4 穩定，fold 7 尚在學習。最佳 checkpoint 避免保存 fold 1 後期退化的權重。

## Artifacts 與可重複性

六個 v2 runs 均已備份到：

`/content/drive/MyDrive/urbansound8k_data/experiment_artifacts/pretrained_cnn_transfer/`

每個 run 包含 folds 1、4、7 的設定、history、validation metrics、manifest 與 `best_model.pt`。已核對每個 Drive run 都有三個 checkpoint，summary 與 Colab 本地結果一致，且 `test_evaluated=false`。總實驗 wall time 約 1,100.6 秒（18 分 20.6 秒）。Dataset、cache、checkpoints、results 與 figures 不提交 GitHub。

## 最終決策與下一步

1. 鎖定 `configs/pretrained_cnn_v2_mixup_last3.yaml` 的方法參數，不再依單一 fold 結果調整。
2. 不對 v2 執行新的 fold 10 test。V1 的 `0.9041` 只屬於 v1 唯一封存 test，不能當作 v2 test result。
3. 下一步若執行正式 10-fold，需建立 fixed-config runner：每個 test fold 都排除於選模，並只用其餘 folds 建立 validation split。
4. 論文應同時報告 mean/std、每 fold 結果、額外 AudioSet 資訊及 validation-only 調整流程，避免把 development 改善誤述為 test improvement。
