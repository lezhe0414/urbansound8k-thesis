# Pretrained CNN bold breakthrough study

更新日期：2026-08-17  
狀態：第一階段已完成；第二階段 cross-scale 多 seed 穩健性測試已預註冊、待執行

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

2026-08-17 在 Colab A100 執行 run `pretrained_cnn_bold_breakthrough_b6848c1_20260817_130915`，約 30 分鐘完成。所有數字均為 folds 1、4、7 的 validation mean/std；fold 10 沒有執行。

### Seed-42 單模型初篩

| 模型 | Macro F1 | Accuracy | 判讀 |
| --- | ---: | ---: | --- |
| MN20 control | `0.89128 ± 0.00880` | `0.88697 ± 0.01207` | 同次實驗參考 |
| MN20 + BN freeze | `0.88738 ± 0.01302` | `0.88344 ± 0.00470` | mean 下降，不保留 |
| MN30 | `0.87767 ± 0.01029` | `0.87786 ± 0.01523` | 明顯下降，不保留 |
| MN40 | `0.89543 ± 0.00687` | `0.89101 ± 0.00999` | 達預註冊門檻，進入三 seed 擴展 |

增加寬度並非單調改善：MN30 比 MN20 差，而 MN40 在 seed 42 提高約 `0.00415`。固定 BatchNorm running statistics 也沒有改善 MN20，顯示小 batch 的 running-statistics drift 不是此設定的主要限制。

### 跨尺度 probability ensemble

| 成員 | Macro F1 | Accuracy |
| --- | ---: | ---: |
| MN20 + MN30 | `0.89362 ± 0.01065` | `0.89172 ± 0.01300` |
| MN20 + MN40 | **`0.90128 ± 0.00982`** | **`0.89725 ± 0.01626`** |
| MN30 + MN40 | `0.89199 ± 0.00627` | `0.88873 ± 0.01097` |
| MN20 + MN30 + MN40 | `0.90062 ± 0.01320` | `0.89636 ± 0.01648` |

MN20 + MN40 是本研究的 seed-42 screen winner。其 Macro F1 比同次 MN20 control 高 `0.01000`，也比歷史 focal 三 seed ensemble `0.89395` 高 `0.00733`。這證明跨尺度模型存在互補錯誤，但它仍只是一組固定 seed 的 post-formal development result，不能直接視為穩健的新最佳方法。

### 三 seed 穩健性檢查

MN40 單模型達門檻後，依預註冊規則補跑 seeds 42、123、2026，並平均三模型 softmax probabilities：

| 方法 | Macro F1 | Accuracy |
| --- | ---: | ---: |
| MN40 3-seed ensemble | `0.89440 ± 0.00651` | `0.89060 ± 0.01188` |

此結果只比歷史 MN20 focal 三 seed ensemble `0.89395 ± 0.00932` 高約 `0.00045`。雖然 fold 間標準差較低，但改善量小到不足以排除實驗噪音；較寬 backbone 的 seed-42 優勢沒有在三 seed 集成後保留。因此不把 MN40 或 MN20 + MN40 升級為正式方法，也不重新執行 fold 10。

## 完整性與備份

- Git commit：`b6848c1`。
- 本地 Colab 輸出與 Drive 各有 33 份 `experiment_manifest.json`。
- 所有 manifests 均維持 `test_evaluated=false`。
- Drive：`/content/drive/MyDrive/urbansound8k_data/experiment_artifacts/pretrained_cnn_bold_breakthrough_b6848c1_20260817_130915/`。
- Dataset、waveform cache、checkpoints、results 與 figures 均未提交 GitHub。

## 結論

本研究找到 Macro F1 超過 `0.90` 的探索性 development 候選，但三 seed 檢查沒有支持穩健突破。論文可報告「跨尺度 MN20 + MN40 在固定 development screen 達 `0.90128`，但 MN40 三 seed ensemble 只達 `0.89440`，故未改變正式模型」，並將完整 cross-scale 多 seed 驗證列為 future work。正式結論仍使用鎖定的 MN20 10-fold 結果 `0.87686 ± 0.04048`。

## 第二階段預註冊：完整 cross-scale 多 seed 集成

為直接檢查 seed-42 MN20 + MN40 的改善能否跨 seeds 保留，第二階段固定使用 MN20 與 MN40、seeds 42/123/2026、development folds 1/4/7。每個 validation fold 對六個模型的 softmax probabilities 做等權平均；不搜尋權重、不刪除較弱 seed，也不依單一 fold 選成員。除補訓缺少的 MN20 seeds 123/2026 外，其餘 checkpoints 全部重用第一階段 artifacts。

主要結果在執行前固定為六模型集成的 mean validation Macro F1。若高於歷史 MN20 focal 三 seed ensemble `0.893950613`，視為 cross-scale 多樣性得到 multi-seed 支持；若同時至少達 seed-42 screen `0.901280552`，才稱為重現固定-seed 的探索性突破。三個 same-seed MN20 + MN40 pair 只作穩健性診斷，不用於挑選最佳 seed。Fold 10 繼續封存，無論結果如何都不由本 runner 執行 test evaluation。

執行命令：

```text
python3 scripts/run_pretrained_cnn_bold_multiseed_cross_scale.py \
  --base-output-name pretrained_cnn_bold_breakthrough_b6848c1_20260817_130915 \
  --output-name <unique-run-name> \
  --backup-root <new-google-drive-artifact-directory>
```
