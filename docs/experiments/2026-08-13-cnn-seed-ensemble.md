# CNN 3-seed probability ensemble protocol

- 日期：2026-08-13
- 基礎設定：`configs/cnn_aug_final.yaml`
- Seeds：42、123、2026
- 主要選模指標：validation Macro F1
- Aggregation：三個模型 softmax probability 的算術平均
- EMA：關閉

## 實驗規範

三個 CNN 使用完全相同的模型、資料切分、augmentation、optimizer、scheduler 與 epochs，只改 random seed。每個 seed 都只依 validation Macro F1 保存自己的最佳 checkpoint，不比較三個 seed 後挑出單一模型，也不對個別 seed 執行 test evaluation。

完成三個 validation-only runs 後，先在 validation fold 平均三個 checkpoint 的預測機率並記錄 ensemble validation metrics。只有在 ensemble 方法與三個固定 seeds 已鎖定後，才允許使用 `--run-test` 對三模型 probability ensemble 執行唯一一次 fold 10 test evaluation。

## 執行方式

只訓練三個 seeds 並計算 validation ensemble：

```text
python3 -m src.ensemble \
  --config configs/cnn_aug_final.yaml \
  --fold 10 \
  --seeds 42 123 2026
```

確認設定鎖定後，續用既有 checkpoint 並執行唯一一次 test：

```text
python3 -m src.ensemble \
  --config configs/cnn_aug_final.yaml \
  --fold 10 \
  --seeds 42 123 2026 \
  --skip-existing \
  --run-test
```

若 ensemble test metrics 已存在，程式會拒絕再次讀取 test fold。輸出保存在 `results/cnn_aug_final_3seed_fold10/`，混淆矩陣保存在 `figures/cnn_aug_final_3seed_fold10_confusion_matrix.png`。

## EMA 實驗結論

先前 validation-only EMA 實驗中，EMA 最佳 Macro F1 為 0.76515，一般權重各 epoch 的最佳 Macro F1 為 0.76426，差異僅約 0.00089；validation Accuracy 相同。此幅度不足以證明 EMA 帶來實質改善，因此正式 ensemble 使用一般 online checkpoint，EMA 保留為已完成但未採用的實驗。

## 正式執行結果

- 訓練開始：2026-08-13 21:45:44 UTC
- 訓練結束：2026-08-13 22:01:16 UTC
- 三個 seeds 總訓練時間：15 分 32 秒
- 最終 ensemble test 推論時間：12 秒
- Colab commit：`9b6adb2`

### 各 seed validation 結果

| Seed | 最佳 epoch | Validation Accuracy | Validation Macro F1 | Validation loss |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 9 | 0.7617 | 0.7868 | 0.7406 |
| 123 | 10 | 0.7308 | 0.7618 | 0.8222 |
| 2026 | 5 | 0.7468 | 0.7737 | 0.7548 |

個別 seed 均未執行 test evaluation。每個 checkpoint 都只依自己的 validation Macro F1 選出。

### Validation probability ensemble

| 指標 | 數值 |
| --- | ---: |
| Accuracy | 0.7388 |
| Macro precision | 0.7747 |
| Macro recall | 0.7738 |
| Macro F1 | 0.7699 |
| Loss | 0.6592 |

三模型平均後的 validation Macro F1 `0.7699` 低於最佳單一 seed 42 的 `0.7868`，也低於先前受控搜尋鎖定設定的 `0.7924`。因此 validation 證據不支持用 3-seed ensemble 取代目前的單一 CNN。

### 鎖定後唯一一次 fold 10 test

在 seeds、aggregation 與 checkpoint selection 全部鎖定後，才對 ensemble 執行一次 fold 10 test evaluation：

| 指標 | 數值 |
| --- | ---: |
| Accuracy | 0.8411 |
| Macro precision | 0.8533 |
| Macro recall | 0.8543 |
| Macro F1 | 0.8501 |
| Loss | 0.5416 |

此 test 結果僅作鎖定方法的確認，不參與 ensemble 設定選擇。相較受控搜尋鎖定單一 CNN 的 test Macro F1 `0.8536`，ensemble 低約 `0.0035`；相較歷史 CNN test Macro F1 `0.8413`，ensemble 高約 `0.0088`。由於這些都是同一 fold 的結果，不能據此繼續調參或宣稱 ensemble 有穩定泛化優勢。

## 結論與後續決策

3-seed probability ensemble 已完整實作與執行，但本次沒有改善 validation Macro F1，test Macro F1 亦未超越已鎖定的單一 CNN。因此主要 CNN 設定維持 `configs/cnn_aug_final.yaml`、EMA 關閉，不採用 ensemble 作為正式 10-fold 的預設方法。

可能原因是三個模型使用相同架構、資料切分與訓練流程，錯誤高度相關；此外 seed 123 明顯弱於 seed 42，等權平均會稀釋較強模型的預測。這是有價值的負結果，可在論文中作為穩定性實驗簡短報告。下一個主要實驗應是固定單一 CNN 設定的 10-fold cross-validation，以 mean、standard deviation 與 per-class F1 評估泛化，而不是繼續查看 fold 10。

## Google Drive 備份

完整 artifacts 已備份至：

```text
/content/drive/MyDrive/urbansound8k_data/experiment_artifacts/cnn_aug_final_3seed_fold10_20260813_220331
```

備份包含三個 seed 的 checkpoint、設定與 training history、ensemble validation/test metrics、protocol、混淆矩陣及 Git commit 紀錄。大型 artifacts 不提交到 GitHub。
