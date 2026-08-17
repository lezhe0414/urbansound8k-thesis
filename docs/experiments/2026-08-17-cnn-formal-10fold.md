# Locked from-scratch CNN formal 10-fold cross-validation

更新日期：2026-08-17
狀態：已預註冊；待正式執行

## 研究目的

使用已由 validation-only controlled search 鎖定的 `configs/cnn_aug_final.yaml`，執行
UrbanSound8K 官方十 folds 的固定 cross-validation。此實驗只估計泛化表現及 fold
variability，不再選擇模型、資料增強或超參數。

## 鎖定條件

- Config SHA-256：`6831eedade7a0cb6e7d2e2b98d32bd067bcc1c7fe62568a2059ead4fe68b82e4`。
- Model：from-scratch spectrogram CNN。
- Seed：42。
- Epochs：10。
- EMA：disabled。
- Ensemble：不使用。
- 每個 test fold 的 validation fold 固定為下一個官方 fold：1→2、2→3、...、10→1。
- 每個訓練 run 只依 validation Macro F1 保存最佳 checkpoint。
- 十個官方 folds 各作 test 一次；test metrics 不影響任何後續設定。

完整 learning rate、weight decay、augmentation、class weighting、class-aware sampling、
Mixup 與 scheduler 均由上述 hash 鎖定，不設額外 CLI 覆寫。

## 執行與中斷規範

正式 runner 在每個 test evaluation 前寫入 `formal_test_started.json`，完成後才寫入
`formal_test_completed.json`。`--resume` 只會略過具有完整 completion manifest、metrics
與 predictions 的 folds。若 test 已開始但未完成，runner 會停止，禁止自動重測。
訓練若在 test 前中斷，可保存舊目錄後重新訓練，不會修改 raw data 或 Mel cache。

## 報告項目

- Macro F1、Accuracy、Macro Precision、Macro Recall 的 fold mean 與 population std。
- 十個 fold 的個別 metrics。
- Aggregate per-class F1。
- Aggregate predictions 與 confusion matrix。
- 設定 hash、Git commit、seed、test/validation mapping 及 integrity manifests。

## 執行命令

```text
python3 scripts/run_cnn_formal_cross_validation.py \
  --output-name <unique-formal-run-name> \
  --backup-root <google-drive-artifact-root>
```

中斷後只能在相同 commit、config hash 與 output name 下加入 `--resume`。

## 論文定位

此結果是 from-scratch CNN 的主要正式 10-fold 結果。單一 fold 10 的 Macro F1
`0.8536` 僅是先前鎖定後的確認值，不可取代 10-fold mean/std。Pretrained CNN 的正式
10-fold 與 post-formal development ensembles 必須分開呈現，因為後者使用 AudioSet
預訓練與不同研究階段。
