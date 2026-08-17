# Locked from-scratch CNN formal 10-fold cross-validation

更新日期：2026-08-17
狀態：已完成；十個 test folds 各評估一次

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

## 執行完整性

- Run：`cnn_aug_final_formal_10fold_aef4a4f_20260817_1534`。
- Git commit：`aef4a4f93f14a6bda0b26ab1506c8bacfb26f525`。
- Colab accelerator：NVIDIA A100-SXM4-40GB。
- Config hash：`6831eedade7a0cb6e7d2e2b98d32bd067bcc1c7fe62568a2059ead4fe68b82e4`。
- `formal_test_started.json`：10。
- `formal_test_completed.json`：10。
- `model_selection_used_test_metrics`：`false`。
- Aggregate predictions、summary 與 confusion matrix：已產生。
- Drive backup：10 個 completed run directories；Drive 與 Colab `summary.json` SHA-256 相符。

Drive 位置：
`/content/drive/MyDrive/urbansound8k_data/experiment_artifacts/cnn_aug_final_formal_10fold_aef4a4f_20260817_1534/`。

## 正式 10-fold 結果

| Metric | Mean ± population standard deviation |
| --- | ---: |
| Macro F1 | **`0.79041 ± 0.04755`** |
| Accuracy | `0.77423 ± 0.05431` |
| Macro Precision | `0.79555 ± 0.04770` |
| Macro Recall | `0.79628 ± 0.04887` |

將十個 folds 的 predictions 合併後，aggregate Macro F1 為 `0.79091`，Accuracy 為
`0.77279`。Fold mean 與 aggregate 指標不同是因為各 fold 樣本數不同；論文主要報告
fold mean ± standard deviation。

| Test fold | Validation fold | Examples | Macro F1 | Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 873 | `0.77793` | `0.75258` |
| 2 | 3 | 888 | `0.75084` | `0.72973` |
| 3 | 4 | 925 | `0.70436` | `0.66378` |
| 4 | 5 | 990 | `0.75085` | `0.74545` |
| 5 | 6 | 936 | `0.83917` | `0.83547` |
| 6 | 7 | 823 | `0.80263` | `0.77278` |
| 7 | 8 | 838 | `0.77173` | `0.77446` |
| 8 | 9 | 806 | `0.79131` | `0.77667` |
| 9 | 10 | 816 | `0.85695` | `0.84069` |
| 10 | 1 | 837 | `0.85835` | `0.85066` |

Aggregate per-class F1：

| Class | F1 |
| --- | ---: |
| air conditioner | `0.61010` |
| car horn | `0.90196` |
| children playing | `0.81370` |
| dog bark | `0.87443` |
| drilling | `0.74823` |
| engine idling | `0.68759` |
| gun shot | `0.93782` |
| jackhammer | `0.64812` |
| siren | `0.84013` |
| street music | `0.84702` |

最大 off-diagonal error counts 為 air conditioner→engine idling `160`、
jackhammer→air conditioner `159`、engine idling→air conditioner `132`、
engine idling→jackhammer `131`，以及 jackhammer→drilling `114`。這些結果顯示主要
限制集中在持續性機械聲與衝擊式機械聲之間，而非所有類別平均失效。

## 訓練行為與解讀

各 fold 最佳 checkpoint 出現在 epoch 8--10，平均 epoch `9.1`。最佳 checkpoint 的
training Macro F1 平均為 `0.82670`，clean validation Macro F1 平均為 `0.79833`，平均
差距 `0.02837`。由於 training metrics 是在 Mixup、SpecAugment 及 class-aware sampling
作用下計算，不能直接當成 clean training-set performance。部分 folds 的 validation F1
甚至高於 training F1，因此沒有證據支持「所有 folds 均嚴重 overfit」；較明顯的問題是
fold variability，尤其 fold 3 的 Macro F1 僅 `0.70436`。

正式 mean 比先前單一 fold 10 結果 `0.8536` 低約 `0.0632`，證明單一 fold 會高估此
設定的整體泛化能力。依預註冊規則，此 10-fold 結果只用於估計性能，不再反向調參。
