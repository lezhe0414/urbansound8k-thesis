# Pretrained CNN nested stacking breakthrough study

更新日期：2026-08-17  
狀態：已預註冊；待 development-only 執行

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
