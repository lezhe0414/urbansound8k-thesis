# Pretrained CNN nested stacking breakthrough study

更新日期：2026-08-17  
狀態：已完成；未採用

## 研究問題

MN20 與 MN40 各三 seeds 的等權 probability ensemble 已在 folds 1、4、7 達到
Macro F1 `0.90104 ± 0.00920`。本研究檢查一個受正則化的 meta-classifier 是否能利用
六模型的 class-specific confidence pattern，在不讀取 fold 10 的條件下進一步提高
development Macro F1。

## 固定協定

外層使用 development folds 1、4、7 的 leave-one-fold-out evaluation。對每個 outer
target fold，target labels 不得參與 meta-model fitting 或 regularisation selection；另外
兩個 folds 的 out-of-fold predictions 作為 meta-training data。

Meta-features 是六個固定模型各自的十類 clipped log probabilities，共 60 維；每個
inner fit 只用 training portion 估計 standardisation。分類器固定為 multinomial-capable
logistic regression、L2 penalty、`class_weight=balanced`、`solver=lbfgs`、
`max_iter=2000`、random seed 42。唯一搜尋變因是 `C`，預先固定為
`[0.01, 0.1, 1.0, 10.0]`。對兩個 outer-training folds 做 reciprocal inner validation，
以 mean inner Macro F1 選 `C`；平手時選較小的 `C`。選定後使用兩 folds 重新 fitting，
再對 outer target fold 評估一次。

來源 checkpoints 固定為：

- MN20 seeds 42、123、2026；
- MN40 seeds 42、123、2026；
- 每個模型使用對應 validation fold 的 `best_model.pt`；
- 官方 32 kHz EfficientAT waveform/log-Mel frontend；
- 無 TTA。

## 成功與停止判準

唯一主要比較是 nested-stacking mean validation Macro F1 與同一批 predictions 的六模型
等權 probability mean。只有 stacking mean 高於 baseline，才稱為 development 改善。
Accuracy、inner scores 與單 fold 結果只作輔助分析。這一輪不搜尋 feature transform、
class weighting、member subset 或 per-class threshold；無論結果如何均停止此 stacking
設定，不依 fold 10 決定下一步。

## 完整性限制

- Study type：post-formal development-only exploratory analysis。
- Selection metric：mean validation Macro F1。
- Fold 10：封存；runner 不提供 test CLI 或 test-evaluation path。
- 來源 checkpoints、raw audio、Mel cache 與 waveform cache：唯讀。
- 大型 predictions、figures 與 artifacts：只保存 Colab/Drive，不提交 GitHub。
- 正式 MN20 10-fold Macro F1 `0.87686 ± 0.04048` 不因本研究而改寫。

## 預定命令

```text
python3 scripts/run_pretrained_cnn_stacking_study.py \
  --base-output-name pretrained_cnn_bold_breakthrough_b6848c1_20260817_130915 \
  --source-summary-name pretrained_cnn_bold_multiseed_cross_scale_6237825_20260817_141117 \
  --output-name <unique-run-name> \
  --backup-root <new-google-drive-artifact-directory>
```

## 執行結果

實驗使用 commit `c1e734c`，run name 為
`pretrained_cnn_nested_stacking_c1e734c_20260817_150141`。同一批六模型 predictions
成功重現等權平均基準 Macro F1 `0.90104 ± 0.00920`、Accuracy
`0.89732 ± 0.01416`。Nested stacking 的 Macro F1 為
`0.87438 ± 0.01625`、Accuracy 為 `0.87589 ± 0.01248`，相對基準差
`-0.02666`，未達成功判準。

| Outer fold | Selected C | Equal-average F1 | Stacking F1 | Equal-average accuracy | Stacking accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.01 | 0.90001 | 0.89679 | 0.88202 | 0.89118 |
| 4 | 0.10 | 0.91279 | 0.85877 | 0.91616 | 0.86061 |
| 7 | 0.10 | 0.89032 | 0.86758 | 0.89379 | 0.87589 |

Stacking 的 Macro Precision 為 `0.88171 ± 0.01019`，Macro Recall 為
`0.87774 ± 0.02033`；等權平均分別為 `0.90654 ± 0.01249` 與
`0.90251 ± 0.00483`。退步主要來自 fold 4，但三個 outer folds 均沒有穩健勝過
等權平均。

## 判讀與決策

Meta-model 每次只能從兩個 development folds 學習，卻要估計 60 維模型信心特徵與
十類決策邊界。UrbanSound8K 的 fold 間聲學分布差異使這個二階模型容易學到
fold-specific confidence pattern；balanced class weighting 也沒有抵消此 domain shift。
因此不再搜尋其他 `C`、member subset 或 stacking 變體，避免在三個 development folds
上形成新的過擬合迴圈。

六模型等權 probability ensemble `0.90104 ± 0.00920` 保留為論文中的 post-formal
exploratory development result。正式 MN20 10-fold Macro F1 `0.87686 ± 0.04048` 仍是
可作正式比較的結果，兩者不可混寫。Nested stacking 不執行 fold 10 test，也不取代
任何已鎖定方法。

## 完整性與 artifacts

- `test_evaluated=false`；程式沒有 test CLI。
- 搜尋輸出中沒有 fold 10 prediction path 或 fold 10 labels。
- 18 份 member validation predictions 已產生並用於三個 outer folds。
- Colab output：`results/pretrained_cnn_nested_stacking_c1e734c_20260817_150141/`。
- Google Drive backup：`experiment_artifacts/pretrained_cnn_nested_stacking_c1e734c_20260817_150141/`。
- GitHub 只保存程式、測試與文件；predictions、meta-model artifacts 和 figures 不提交。
