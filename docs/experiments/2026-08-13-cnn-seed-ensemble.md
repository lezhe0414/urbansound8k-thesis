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
