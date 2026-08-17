# Pretrained CNN bold breakthrough study

更新日期：2026-08-17  
狀態：已預註冊，等待 Colab development-only 執行

## 目的與邊界

本研究是在 MN20 正式 10-fold 完成後建立的探索性分支。正式結果已固定為 Macro F1 `0.87686 ± 0.04048`；其 test folds 已被觀察，後續任何改動不得回頭改寫正式結論。本研究只使用 folds 1、4、7 的平均 validation Macro F1 選模，fold 10 完全封存，所有 manifests 必須維持 `test_evaluated=false`。

歷史 development 參考值如下：

- MN20 last-2 single-seed control：Macro F1 `0.890685550`；
- MN20 focal loss + fixed seeds 42/123/2026 probability ensemble：Macro F1 `0.893950613 ± 0.009315`。

## 假設

1. **Backbone capacity**：官方 AudioSet MN30/MN40 比 MN20 更寬，可能提高可轉移 representation 的上限，但也可能增加 UrbanSound8K 小資料集上的 variance。
2. **BatchNorm stability**：partial fine-tuning 時固定 encoder BatchNorm running statistics，可能避免小 batch 破壞 AudioSet 預訓練統計。
3. **Cross-scale diversity**：MN20、MN30、MN40 的機率平均可能利用不同尺度的互補錯誤，且不需要讀取 test fold。

## 固定條件

| 項目 | 值 |
| --- | --- |
| Development folds | 1, 4, 7 |
| Sealed test fold | 10 |
| Initial screen seed | 42 |
| Linear-probe epochs | 5 |
| Partial fine-tune epochs | 8 |
| Unfrozen blocks | last 2 |
| Encoder/head LR | `2e-5` / `3e-4` |
| Loss | class-balanced focal, gamma `1.5` |
| Mixup | alpha `0.15`, probability `0.5` |
| Selection metric | mean validation Macro F1 |
| TTA | off |

四個 seed-42 候選為 MN20 control、MN20 BatchNorm freeze、MN30 與 MN40。MN30/MN40 checkpoint 分別為官方 `mn30_as_mAP_482.pt` 與 `mn40_as_mAP_484.pt`，沿用 pinned EfficientAT upstream commit 與 MIT License。所有模型使用相同官方 32 kHz waveform 與 log-Mel frontend。

## 預先決定的擴展規則

先完成四個單模型及七個跨尺度 probability ensembles。新單模型只有在 mean validation Macro F1 至少達 `0.890685550` 時，才擴展至固定 seeds 42、123、2026。若沒有新單模型達門檻，停止多 seed 訓練；不得因接近 test 結果而放寬門檻。

## 執行命令

```text
python3 scripts/run_pretrained_cnn_bold_study.py \
  --backup-root /content/drive/MyDrive/urbansound8k_data/experiment_artifacts/pretrained_cnn_bold_breakthrough
```

只做 seed-42 初篩：

```text
python3 scripts/run_pretrained_cnn_bold_study.py \
  --screen-only \
  --backup-root /content/drive/MyDrive/urbansound8k_data/experiment_artifacts/pretrained_cnn_bold_breakthrough
```

## 輸出

每個 fold 保存 resolved config、history、validation metrics、best checkpoint、manifest 與 training-history figure；跨尺度 evaluation 保存 validation predictions、metrics 與 confusion matrix。摘要寫入 `results/pretrained_cnn_bold_breakthrough/`，並即時備份到上述 Drive 目錄。Dataset、waveform cache、checkpoints、results 與 figures 不提交 GitHub。

## 結果

待 Colab 執行後回填。任何勝出結果都只能稱為 post-formal development-only exploratory evidence，不得取代正式 10-fold 結果。
