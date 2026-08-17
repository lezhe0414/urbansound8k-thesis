# EfficientAT pretrained CNN v2 development study

更新日期：2026-08-17
狀態：實作中；fold 10 不再評估

## 研究目的

在 v1 已鎖定並完成一次 fold 10 final confirmation 後，本研究只使用 folds 1、4、7 的 validation Macro F1，評估更長訓練、gradual unfreezing、輕量 pretrained-specific augmentation 與解凍深度。V1 的 fold 10 Macro F1 `0.9041` 不參與 v2 選模，v2 也不得再次執行單一 fold 10 test。

V1 development reference 為 Macro F1 `0.8716 ± 0.0283`、Accuracy `0.8734 ± 0.0212`。每組實驗保持 seed 42、相同資料切分、官方 32 kHz EfficientAT frontend、class weighting、class-aware sampling、optimizer 與 differential learning rates不變。

## 受控序列

| 階段 | Run name | 唯一變因 | Development Macro F1 |
| --- | --- | --- | ---: |
| V1 reference | `pretrained_cnn_mn10_partial_ft_lr2e5_v1` | 5 epochs、last 2 blocks | 0.8716 ± 0.0283 |
| A | `pretrained_cnn_mn10_v2_epochs8` | epochs 5 → 8 | 待執行 |
| B | `pretrained_cnn_mn10_v2_gradual8_head2` | 前 2 epochs head-only，再解凍 last 2 | 待執行 |
| C | `pretrained_cnn_mn10_v2_mask_f8_t24` | official frontend frequency/time masks 8/24 | 待執行 |
| D | `pretrained_cnn_mn10_v2_mixup_a015_p050` | post-frontend Mixup alpha 0.15、probability 0.5 | 待執行 |
| E | 待 A–D 勝出後建立 | 比較 last 1/2/3 blocks | 待執行 |

Stage A 是其餘 v2 candidates 的共同 8-epoch control。Stages B–D 各自只改一類變因，不互相疊加。完成 A–D 後，選擇 mean validation Macro F1 最高的設定作為解凍深度比較基礎；目前的 last 2 結果直接作為 E 的中間點，只額外執行 last 1 與 last 3。

## Augmentation 邊界

- Frequency/time masking 只在模型為 training mode 時由 EfficientAT 官方 `AugmentMelSTFT` 執行；validation frontend 固定為 evaluation mode。
- Mixup 在官方 log-Mel frontend 之後執行，不改變 waveform cache 或原始音訊。
- Mixup training Accuracy/F1 只描述混合 batch 上相對原始標籤的 proxy，不用於選模；唯一選模指標仍為未增強 validation Macro F1。
- 不重新 preprocessing，不覆寫 `data/raw`、既有 Mel cache或 waveform cache。

## 停止與評估規則

1. 每個 candidate 都執行 folds 1、4、7，並即時備份到 Google Drive。
2. 只比較三-fold validation Macro F1 mean/std；Accuracy mean/std 只作輔助。
3. 沒有改善的 candidate 不疊加到後續設定。
4. V2 不執行 fold 10 test。若鎖定新設定，下一步是正式固定設定 cross-validation，而不是再次查看 fold 10。
5. 大型 cache、checkpoints、results 與 figures 不提交 GitHub。
