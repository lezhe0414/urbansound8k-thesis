# EfficientAT recommended development and formal validation study

更新日期：2026-08-17  
狀態：程式已實作，development 實驗待執行；formal test 尚未啟動

## 目的

在不反覆調整既有 learning rate 或 dropout 的前提下，評估四種延伸：waveform-level augmentation、time-shift test-time augmentation (TTA)、固定三 seed probability ensemble，以及較寬的 AudioSet-pretrained EfficientAT MN20。所有方法先只在 folds 1、4、7 的 development validation 上比較，唯一主要指標為平均 validation Macro F1，fold 10 在方法鎖定前保持不可見。

## 固定條件

- Dataset：UrbanSound8K。
- Development validation folds：1、4、7。
- Sealed fold：10；development 階段禁止 test evaluation。
- Base model：EfficientAT MN10 `mn10_as_mAP_471.pt`。
- Official frontend：32 kHz mono waveform、5 秒、128-bin log-Mel、官方 normalization。
- Base fine-tuning：8 epochs、seed 42、last 3 convolution blocks、encoder/head LR `2e-5`/`3e-4`。
- Balancing：class-weighted loss 與 class-aware sampling power `0.5`。
- Post-frontend Mixup：alpha `0.15`、probability `0.5`。
- Checkpoint selection：每 fold 只依 validation Macro F1。

既有 v2 reference 為 Macro F1 `0.8844 ± 0.0165`、Accuracy `0.8824 ± 0.0141`。

## Stage 1: waveform augmentation

所有 waveform augmentation 都在既有 cache 載入後、官方 frontend 前動態執行，不修改 raw audio 或 cache。

| 候選 | 唯一方法差異 |
| --- | --- |
| Shift + gain | 每樣本 0.5 機率做最多 ±0.5 秒零填補平移；0.5 機率做 ±3 dB gain |
| Gaussian noise | 每樣本 0.35 機率加入 25--40 dB SNR Gaussian noise |
| Combined | Shift/gain 同上，另以 0.25 機率加入 25--40 dB SNR noise |

三組使用相同 seed、folds、epochs 與優化器。只有平均 validation Macro F1 高於 v2 reference 且 fold variance沒有明顯惡化時才保留。

## Stage 2: TTA

TTA 不重新訓練模型。對勝出 checkpoint 產生 `-0.5`、`0`、`+0.5` 秒三個零填補時間位置，平均 softmax probabilities。比較使用同一批 checkpoint 的無 TTA 與 TTA validation Macro F1。

## Stage 3: fixed three-seed ensemble

使用 seeds 42、123、2026，各自重新執行 linear probing 與相同的 fixed partial fine-tuning。每個 validation fold 對三個 checkpoint 的 probabilities 作算術平均。不得挑選三個 seeds 中表現最高者；ensemble 必須以預先固定的三個 seeds 整體判斷。

## Stage 4: stronger pretrained CNN

使用 EfficientAT MN20 `mn20_as_mAP_478.pt`，其官方 AudioSet mAP 高於 MN10 的 0.471。MN20 與 MN10 使用相同官方 frontend、MIT-licensed upstream implementation、五 epoch linear probing及八 epoch last-three-block fine-tuning。第一個 MN20 比較不加入 waveform augmentation，避免同時改變模型容量與 augmentation。

## Formal 10-fold gate

Development 完成後只鎖定一個唯一方法。正式 10-fold 使用固定 cyclic mapping：test fold 1 對應 validation fold 2，依此類推，test fold 10 對應 validation fold 1。每個 test fold 從訓練與 checkpoint selection 完全排除，且只評估一次。若 development 勝出的是三 seed ensemble，正式 10-fold 亦固定使用三個 seeds；否則使用單一 seed。正式輸出包含 mean/std、per-class F1、每 fold 與 aggregate confusion matrix。

## Artifact locations

- Local/Colab outputs：`results/`
- Drive backup：`/content/drive/MyDrive/urbansound8k_data/experiment_artifacts/pretrained_cnn_recommended/`
- GitHub：只提交 source、configs、tests 與 documentation，不提交 dataset、cache、checkpoints、results 或 figures。

