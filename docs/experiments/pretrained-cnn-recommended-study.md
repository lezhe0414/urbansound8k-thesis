# EfficientAT recommended development and formal validation study

更新日期：2026-08-17  
狀態：development 已完成，唯一方法已鎖定；formal 10-fold 尚未啟動

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

## Development results

| 方法 | Validation Macro F1 mean ± std | Validation Accuracy mean ± std | 決策 |
| --- | --- | --- | --- |
| MN10 v2 control，seed 42 | `0.88437 ± 0.01650` | `0.88236 ± 0.01406` | Reference |
| Shift + gain | `0.88131 ± 0.01633` | `0.87900` | Reject |
| Gaussian noise | `0.88065 ± 0.01958` | `0.87939` | Reject |
| Shift + gain + noise | `0.88012 ± 0.01787` | `0.88001` | Reject |
| MN10 three-view time-shift TTA | `0.87907 ± 0.01229` | `0.87894 ± 0.01063` | Reject |
| MN10 fixed 3-seed ensemble | `0.88463 ± 0.01834` | `0.88487 ± 0.01545` | Reject: negligible mean gain and worse variance |
| MN20 linear probe | `0.87581 ± 0.01731` | `0.87454 ± 0.00753` | Continue to partial fine-tuning |
| MN20 partial fine-tuning, last 3 blocks | `0.88587 ± 0.01249` | `0.88386 ± 0.00925` | Competitive; test one adjacent depth |
| **MN20 partial fine-tuning, last 2 blocks** | **`0.89069 ± 0.01165`** | **`0.88742 ± 0.01062`** | **Selected and locked** |

The selected MN20 folds 1, 4 and 7 achieved Macro F1 values of `0.89971`, `0.89811` and `0.87424`, respectively. Its mean gain over the MN10 v2 control is `+0.00631`, and its fold standard deviation is lower by `0.00485`. The fixed MN10 three-seed ensemble improved the mean by only `+0.00026` while increasing variance, so it was not selected.

## Stage 1: waveform augmentation

所有 waveform augmentation 都在既有 cache 載入後、官方 frontend 前動態執行，不修改 raw audio 或 cache。

| 候選 | 唯一方法差異 |
| --- | --- |
| Shift + gain | 每樣本 0.5 機率做最多 ±0.5 秒零填補平移；0.5 機率做 ±3 dB gain |
| Gaussian noise | 每樣本 0.35 機率加入 25--40 dB SNR Gaussian noise |
| Combined | Shift/gain 同上，另以 0.25 機率加入 25--40 dB SNR noise |

三組使用相同 seed、folds、epochs 與優化器。三組平均 Macro F1 均低於 control，因此全部拒絕；此結論不使用任何 test fold 結果。

## Stage 2: TTA

TTA 不重新訓練模型。對勝出 checkpoint 產生 `-0.5`、`0`、`+0.5` 秒三個零填補時間位置，平均 softmax probabilities。無 TTA 重算完全重現 control `0.88437`；TTA 降至 `0.87907`，因此正式方法停用 TTA。

## Stage 3: fixed three-seed ensemble

使用 seeds 42、123、2026，各自重新執行 linear probing 與相同的 fixed partial fine-tuning。每個 validation fold 對三個 checkpoint 的 probabilities 作算術平均。固定 ensemble 達 `0.88463 ± 0.01834`，只比 control 高 `0.00026` 且 variance 較高，因此不承擔正式 10-fold 的三倍訓練成本。

## Stage 4: stronger pretrained CNN

使用 EfficientAT MN20 `mn20_as_mAP_478.pt`，其官方 AudioSet mAP 高於 MN10 的 0.471。MN20 與 MN10 使用相同官方 frontend、MIT-licensed upstream implementation、五 epoch linear probing及八 epoch partial fine-tuning。MN20 linear probe 達 `0.87581`；解凍最後 3 blocks 達 `0.88587`。依預先限制只再測一個鄰近設定，解凍最後 2 blocks 提高至 `0.89069` 且 variance 下降，因此勝出。

## Locked method

唯一正式設定為 `configs/pretrained_cnn_mn20_locked_last2.yaml`：

- EfficientAT MN20 AudioSet checkpoint `mn20_as_mAP_478.pt`；
- 官方 32 kHz、5 秒 waveform frontend；
- seed 42、batch size 32；
- 5-epoch frozen-encoder linear probe，接續 8-epoch partial fine-tuning；
- 僅解凍最後 2 個 convolution blocks；
- encoder/head learning rates `2e-5`/`3e-4`，weight decay `1e-4`；
- class-weighted loss 與 class-aware sampling power `0.5`；
- Mixup alpha `0.15`、probability `0.5`；
- waveform augmentation、frontend masking、gradual unfreezing 與 TTA 均停用；
- checkpoint 只依 validation Macro F1 保存。

此設定的 `evaluation.locked_for_test` 已設為 `true`。鎖定時 fold 10 仍未被本 study 評估。

## Formal 10-fold gate

正式 10-fold 使用固定 cyclic mapping：test fold 1 對應 validation fold 2，依此類推，test fold 10 對應 validation fold 1。每個 test fold 從訓練與 checkpoint selection 完全排除，且只評估一次。因勝出方法是單一 seed MN20，正式執行固定 seed 42、無 TTA，不再比較候選。正式輸出包含 mean/std、per-class F1、每 fold 與 aggregate confusion matrix。

## Artifact locations

- Local/Colab outputs：`results/`
- Drive backup：`/content/drive/MyDrive/urbansound8k_data/experiment_artifacts/pretrained_cnn_recommended/`
- GitHub：只提交 source、configs、tests 與 documentation，不提交 dataset、cache、checkpoints、results 或 figures。
