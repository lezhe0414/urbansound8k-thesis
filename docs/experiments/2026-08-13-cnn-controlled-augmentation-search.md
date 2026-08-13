# CNN 受控資料增強搜尋紀錄

- 日期：2026-08-13
- Search ID：`cnn_aug_20260813_1903`
- 主要選模指標：validation Macro F1
- 輔助指標：validation Accuracy、validation loss
- 固定條件：UrbanSound8K、seed 42、fold 1 validation、fold 10 test、10 epochs
- Google Drive 備份：`/content/drive/MyDrive/urbansound8k_data/cnn_controlled_search_cnn_aug_20260813_1903/`
- 備份核對：97 個檔案，包含設定、history、validation metrics、checkpoint、圖表、搜尋 CSV 與最終報告

## 實驗規範

資料增強只在 training batch 載入 Mel-spectrogram tensor 後即時執行，沒有重新 preprocessing，也沒有修改原始音訊或 8,732 個 `.npz` cache。初始四組設定使用相同 fold、seed、epochs、資料切分與評估方式。後續每輪只改一類變因，只有 validation Macro F1 改善才保留，否則回復上一個最佳設定。連續五輪未改善後停止。fold 10 test 在唯一最佳設定鎖定前保持不可見，最後只評估一次。

## 初始四組比較

| Profile | Validation Macro F1 | Validation Accuracy | 耗時（秒） | 決策 |
| --- | ---: | ---: | ---: | --- |
| control | 0.7635 | 0.7354 | 259.7 | 不保留 |
| light | 0.7474 | 0.7239 | 283.4 | 不保留 |
| balanced | 0.7679 | 0.7411 | 297.1 | 不保留 |
| strong | 0.7800 | 0.7526 | 313.0 | 初始勝出 |

## 單一變因迭代

| 輪次 | 唯一改動 | 新值 | Validation Macro F1 | Validation Accuracy | 耗時（秒） | 決策 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | SpecAugment probability | 0.60 | 0.7617 | 0.7354 | 310.5 | 回復 |
| 2 | Time-mask width | 21 | 0.7686 | 0.7411 | 313.9 | 回復 |
| 3 | Frequency-mask width | 9 | 0.7801 | 0.7526 | 保留 |
| 4 | Batch Mixup probability | 0.45 | **0.7924** | **0.7709** | 312.8 | 保留；最終勝出 |
| 5 | Mixup alpha | 0.10 | 0.7660 | 0.7342 | 310.8 | 回復 |
| 6 | Class-aware sampling power | 0.35 | 0.7651 | 0.7342 | 314.5 | 回復 |
| 7 | Label smoothing | 0.01 | 0.7686 | 0.7423 | 311.2 | 回復 |
| 8 | Learning rate | 0.00035 | 0.7829 | 0.7572 | 315.4 | 回復 |
| 9 | Weight decay | 0.002 | 0.7780 | 0.7503 | 317.2 | 回復 |

第 3 輪的改善只有約 0.00005，極可能落在單次訓練的隨機波動範圍，不應單獨宣稱 frequency mask 9 具有實質效果。第 4 輪的 Mixup probability 調整帶來較明顯提升。第 5 至 9 輪連續未改善，因此依預先規定的 patience 5 停止，沒有執行第 10 輪。13 個 validation runs 的訓練時間約 66.2 分鐘，不含同步、備份與最終 test 時間。

## 唯一最佳設定

- 固定設定檔：`configs/cnn_aug_final.yaml`
- Run：`cnn_aug_cnn_aug_20260813_1903_iter04_mixup_probability`
- Model：CNN，dropout 0.35，spatial dropout 0.05
- Batch size：32；epochs：10；learning rate：0.00045；weight decay：0.001
- Label smoothing：0.03
- Class weighting power：0.5；class-aware sampling power：0.5
- Cosine scheduler，minimum learning rate：0.00001
- SpecAugment probability：0.70；frequency mask：9 x 2；time mask：28 x 2
- Mixup mode；batch-mix probability：0.45；Mixup alpha：0.30
- 其他即時增強：time shift、frequency shift、time stretch、gain、Gaussian noise

最佳 checkpoint 位於 epoch 9。該 epoch 的 train Macro F1 為 0.8255，validation Macro F1 為 0.7924，差距為 0.0331。Control 的對應差距約為 0.1868，因此受控增強顯著縮小了 train-validation gap；不過 Mixup 下的 training metrics 受混合標籤影響，不能與未增強訓練作完全等價的解讀。

## 唯一一次 fold 10 test

| 指標 | 數值 |
| --- | ---: |
| Accuracy | 0.8471 |
| Macro F1 | **0.8536** |
| Macro precision | 0.8618 |
| Macro recall | 0.8601 |
| Test loss | 0.5253 |

相較歷史 CNN fold 10 test（Accuracy 0.8280、Macro F1 0.8413），Accuracy 提高約 0.0191，Macro F1 提高約 0.0123。這是單一 test fold 的差異，不能視為統計顯著結論，也不能再用來調整設定。

## 結論與下一步

Strong profile 是初始四組中最佳者；在其基礎上，降低 batch Mixup probability 至 0.45 是最有證據的有效改動。更低的 Mixup alpha、較弱 class-aware sampling、較低 label smoothing、較低 learning rate 與較高 weight decay 均未改善 validation Macro F1。現設定已鎖定為 `configs/cnn_aug_final.yaml`，值得執行正式 UrbanSound8K 10-fold cross-validation，報告各 fold Macro F1 與 Accuracy 的 mean、standard deviation 及每類表現。10-fold 階段不得再依 fold 結果調參。

論文應公平說明這是一個先比較四種 augmentation profiles、再以 greedy one-variable-at-a-time 方式進行最多十輪的 validation-only 搜尋；搜尋因連續五輪無改善而於第九輪停止。所有模型選擇只依 validation Macro F1，fold 10 test 僅在唯一設定鎖定後使用一次。最終論文的主要結論應以尚待完成的 10-fold 統計結果為準。
