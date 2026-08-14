# CNN Breakthrough Protocol: Targeting Higher Macro F1

日期：2026-08-13（結果於 2026-08-14 完成）

分支：`codex/cnn-breakthrough-90`

狀態：已完成；沒有候選達到可取代穩定主線的證據，因此不合併回 `main`

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

實際執行使用 search ID `20260813_breakthrough_v1`，共完成 5 個候選 x 3 個 development folds，即 15 個 validation-only runs。第一個 run 開始至最後一個 run 結束約 1 小時 21 分鐘。Google Drive 備份位置為：

```text
/content/drive/MyDrive/urbansound8k_data/experiment_artifacts/
cnn_breakthrough_20260813_breakthrough_v1
```

備份經核對包含 15 個 run directories、94 個檔案；結果紀錄中的 `test_evaluated` 為 `false`。本研究沒有讀取或評估 fold 10 test。

## 實驗結果

以下標準差為三個 development folds 的 population standard deviation。排名只使用平均 validation Macro F1；Accuracy 為輔助資訊。

| 排名 | 候選 | Fold 1 F1 | Fold 4 F1 | Fold 7 F1 | Mean F1 | F1 std | Mean Accuracy | Accuracy std |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Augmentation cooldown | 0.7689 | 0.7943 | 0.7830 | 0.7821 | 0.0104 | 0.7740 | 0.0209 |
| 2 | Control | 0.7813 | 0.7767 | 0.7875 | 0.7818 | 0.0044 | 0.7740 | 0.0144 |
| 3 | SE CNN | 0.7625 | 0.7688 | 0.7980 | 0.7764 | 0.0155 | 0.7703 | 0.0209 |
| 4 | Single balancing | 0.7597 | 0.7632 | 0.8008 | 0.7746 | 0.0186 | 0.7626 | 0.0261 |
| 5 | Mel + delta + delta-delta | 0.7766 | 0.7681 | 0.7755 | 0.7734 | 0.0038 | 0.7626 | 0.0071 |

Cooldown 的平均 Macro F1 只比 control 高 `0.00025`，但其 F1 標準差由 `0.0044` 增至 `0.0104`，且改善主要來自 fold 4；fold 1 反而下降。這個差異遠小於不同 folds 間的自然變異，不能視為穩健改善。Control 仍是本次比較中最穩定的設定。

Delta channels 在三個 folds 都低於對應 control，且增加約 15% 執行時間，故不採用。Single balancing 與 SE CNN 偶爾在 fold 7 得到較高分，但平均值較低且折間波動更大，亦不採用。

## 晉級條件

- 候選必須先超越同協定 control 的平均 validation Macro F1。
- 改善不能只來自一個 fold；需同時檢查平均值與標準差。
- 勝出候選可再做有限、單一變因的鄰近實驗。
- 只有唯一設定完成鎖定後，才能另行決定是否值得進行正式 10-fold cross-validation。
- 此突破分支不再重複評估 fold 10；最終論文主結果以未洩漏的正式評估為準。

## 最終決策

本實驗沒有候選同時滿足「平均 Macro F1 明確提高」與「跨 folds 穩定」兩項條件，因此不進行第二階段鄰近調參，也不把突破分支合併回 `main`。`0.90` 目標未達成，但負結果指出目前瓶頸不能只靠增加輸入通道、SE attention、重複平衡或簡單 augmentation cooldown 解決。

下一步回到 `main` 已鎖定的 `configs/cnn_aug_final.yaml`，不再修改設定，直接執行正式 10-fold cross-validation。其 mean/std、per-class F1 與 aggregate confusion matrix 才作為論文中的主要 CNN 泛化證據。

先前受控搜尋的 validation Macro F1 `0.7924` 來自另一個單一 validation split，不能與本研究的三-fold mean `0.7818` 直接作同條件比較。論文應分開描述兩個協定，避免把 split 差異誤述為模型退步。

## 0.90 目標的解讀

`0.90` 是研究目標，不是可保證的結果。若五個候選均未接近該值，代表目前從零訓練 CNN 與既有資料表示可能接近容量上限；下一個高機率方向應是使用符合官方前處理規格的 AudioSet-pretrained AST 或 PANNs，而不是繼續以 fold 10 反覆調參。
