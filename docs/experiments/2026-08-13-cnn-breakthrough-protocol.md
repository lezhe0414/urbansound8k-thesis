# CNN Breakthrough Protocol: Targeting Higher Macro F1

日期：2026-08-13

分支：`codex/cnn-breakthrough-90`

狀態：程式已實作，GPU 實驗待執行

## 研究目的

目前鎖定的單一 CNN 在 development validation 的 Macro F1 為 `0.7924`，唯一一次 fold 10 test Macro F1 為 `0.8536`。本實驗以 `0.90` 作為高難度攻擊目標，但不把達標視為保證，也不使用 fold 10 test 選模型。

既有 `configs/cnn_aug_final.yaml` 保持不變。本實驗使用獨立分支與獨立 run names；候選若沒有在多個 development folds 上穩定改善，不合併回 `main`。

## 資料分割與選模規則

- 鎖定 test fold：UrbanSound8K fold 10，不在候選搜尋期間評估。
- Development validation folds：1、4、7。
- 每個候選使用相同 seed、epochs、optimizer、資料增強與評估方式，只有指定變因不同。
- 主要指標：三個 development folds 的平均 validation Macro F1。
- 輔助指標：validation Accuracy、Macro F1 標準差、最佳 epoch 與耗時。
- 禁止根據 fold 10 test、既有 test 結果或單一特別高的 validation fold 選模型。

此切分會將 fold 10 從所有候選的 training 和 validation 中排除。對某個 development validation fold 訓練時，其餘八個非 test folds 作 training。

## 初始候選

| 候選 | 唯一主要變因 | 設定檔 |
| --- | --- | --- |
| Control | 鎖定 CNN 的 validation-only 對照 | `configs/cnn_breakthrough_control.yaml` |
| Mel + delta + delta-delta | 將 cached Mel 動態組成三通道，不重新 preprocessing | `configs/cnn_breakthrough_delta.yaml` |
| Augmentation cooldown | epochs 6-10 逐步降低 augmentation 與 Mixup 機率 | `configs/cnn_breakthrough_cooldown.yaml` |
| Single balancing | 關閉 class-aware sampler，保留 class-weighted loss | `configs/cnn_breakthrough_single_balance.yaml` |
| SE CNN | 四層卷積骨幹加入 squeeze-and-excitation channel attention | `configs/cnn_breakthrough_se.yaml` |

## 為何選這些候選

1. Delta channels 顯式提供頻譜隨時間的一階與二階變化，可能改善 siren、engine idling 等動態聲音的辨識。
2. Cooldown 讓前期強增強建立不變性，後期回到較接近真實資料的分布以細化決策邊界。
3. 同時使用 weighted sampler 與 weighted loss 可能重複補償類別不平衡；single balancing 可檢查是否因此犧牲整體校準。
4. SE attention 增加模型對重要 channel features 的選擇能力，但仍維持比大型 pretrained model 更低的運算成本。

## 執行與保存

```text
python3 scripts/run_cnn_breakthrough_search.py \
  --search-id 20260813_breakthrough_v1 \
  --backup-root /content/drive/MyDrive/urbansound8k_data/experiment_artifacts
```

Runner 每完成一個 candidate/fold 便立即寫入 progress CSV/JSON，保存 config、history、validation metrics、checkpoint 與必要圖表，並同步到指定 Drive 目錄。`metrics.json` 若在候選搜尋中出現，流程會視為 test protocol 違規並停止。

## 晉級條件

- 候選必須先超越同協定 control 的平均 validation Macro F1。
- 改善不能只來自一個 fold；需同時檢查平均值與標準差。
- 勝出候選可再做有限、單一變因的鄰近實驗。
- 只有唯一設定完成鎖定後，才能另行決定是否值得進行正式 10-fold cross-validation。
- 此突破分支不再重複評估 fold 10；最終論文主結果以未洩漏的正式評估為準。

## 0.90 目標的解讀

`0.90` 是研究目標，不是可保證的結果。若五個候選均未接近該值，代表目前從零訓練 CNN 與既有資料表示可能接近容量上限；下一個高機率方向應是使用符合官方前處理規格的 AudioSet-pretrained AST 或 PANNs，而不是繼續以 fold 10 反覆調參。
