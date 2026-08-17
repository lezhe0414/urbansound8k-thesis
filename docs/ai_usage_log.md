# AI Agent 使用紀錄

這份文件用來記錄 AI Agent 在論文與程式專案中協助了哪些工作。教授已同意使用 AI Agent，但仍建議保留紀錄，以便日後需要向教授、學校或口試委員說明使用範圍。

## 使用原則

- 記錄重要協作，不需要記錄每一次微小修字。
- 若 AI 協助產生論文文字、程式碼、圖表、文獻整理或實驗解讀，應留下紀錄。
- AI 產出的內容仍需由使用者確認、修改與負責。
- 不把未驗證的 AI 推論當成已證實事實。
- 若學校或教授有指定揭露格式，以其要求為準。

## 使用紀錄模板

| 日期 | AI 協助內容 | 涉及檔案 / 成果 | 使用者後續確認 | 備註 |
| --- | --- | --- | --- | --- |
| 2026-06-28 | 建立論文協作專案骨架、文件模板、程式與實驗管理流程 | `README.md`、`AGENTS.md`、`docs/`、`src/`、`references/` | 待使用者確認後續研究內容 | 教授已同意使用 AI Agent |
| 2026-06-29 | 協助下載並驗證 UrbanSound8K，執行 Mel-spectrogram preprocessing，跑 CNN 與 Transformer smoke experiments，完成 Transformer fold 10 正式訓練與評估，並更新可重複執行設定 | `data/raw/UrbanSound8K_soundata/`、`data/processed/urbansound8k_mels/`、`configs/`、`src/`、`results/`、`figures/` | 待使用者確認 CNN 正式長訓練環境與教授回饋 | 資料與結果輸出未提交到 git |
| 2026-07-02 | 協助建立 Google Colab CNN baseline 執行 notebook，並用英文註解說明 GitHub 同步、資料下載、preprocessing、訓練、評估與結果打包流程 | `notebooks/2026-07-02-colab-cnn-baseline.ipynb`、`docs/progress_tracker.md`、`docs/artifact_index.md` | 待使用者於 Colab 執行並下載結果 | Colab 用於 GPU 訓練；GitHub 仍作為程式碼來源 |
| 2026-07-08 | 協助新增 Google Colab CNN + Transformer fold 10 notebook，整理 Drive cache、CNN baseline、Spectrogram Transformer、metrics 與 artifacts 打包流程 | `notebooks/2026-07-08-colab-cnn-transformer-fold10.ipynb`、`docs/progress_tracker.md`、`docs/artifact_index.md` | 待使用者於 Colab 執行並回填 CNN/Transformer metrics | 大型資料與實驗輸出仍不提交到 git |
| 2026-08-13 | 設計並實作 CNN Mel-spectrogram data augmentation ablation，包括逐樣本增強、Mixup/CutMix、validation-only model selection、測試與 Colab 執行流程 | `src/data/augmentation.py`、`src/train.py`、`configs/cnn_aug_*.yaml`、`scripts/run_cnn_augmentation_ablation.py`、`tests/test_augmentation.py` | 待使用者於 Colab 執行四組正式 fold 10 實驗並確認結果 | 避免以 test fold 調參；正式數據尚未產生 |
| 2026-08-13 | 將 CNN augmentation ablation 擴充為受控、自動復原的 validation-only 搜尋流程，加入唯一 run name、逐輪紀錄、一次重試、早停、training history 圖及 Google Drive 即時備份 | `scripts/run_cnn_controlled_search.py`、`src/train.py`、`src/utils/plotting.py`、`tests/test_controlled_search.py`、相關流程文件 | 待 Colab 正式實驗完成後由使用者確認結果 | Test 僅在唯一最佳設定鎖定後評估一次 |
| 2026-08-13 | 操作 Colab 完成 CNN 受控資料增強搜尋，依 validation Macro F1 比較四組初始設定與九輪單一變因迭代，鎖定唯一設定後執行一次 fold 10 test，並核對 Google Drive 備份 | `docs/experiments/2026-08-13-cnn-controlled-augmentation-search.md`；Drive 搜尋 artifacts | 使用者需確認是否以鎖定設定執行正式 10-fold | 沒有重新 preprocessing；沒有用 test 結果調參 |
| 2026-08-13 | 實作 CNN EMA 權重追蹤、online/EMA validation 同次比較、checkpoint metadata 與測試，並記錄延後的 3-seed ensemble | `src/utils/ema.py`、`src/train.py`、`configs/cnn_aug_ema.yaml`、`tests/test_ema.py`、狀態與決策文件 | 待使用者在 Colab 執行 validation-only EMA 實驗 | 不改動既有 final config；不再次執行 fold 10 test |
| 2026-08-13 | 操作 Colab 完成 EMA validation 比較與固定 3-seed probability ensemble，核對鎖定協定、整理 validation/test 指標並備份 artifacts 至 Google Drive | `src/ensemble.py`、`configs/cnn_aug_final.yaml`、`docs/experiments/2026-08-13-cnn-seed-ensemble.md`；Drive ensemble artifacts | 使用者需確認是否直接進入固定單一 CNN 的正式 10-fold | EMA 與 ensemble 均未超越鎖定單一 CNN；大型結果不提交 GitHub |
| 2026-08-17 | 選擇並整合 AudioSet-pretrained EfficientAT MN10，建立官方 32 kHz waveform frontend、獨立 cache、linear probing、partial fine-tuning、三-fold validation runner、測試與研究紀錄 | `src/models/pretrained_efficientat.py`、`src/data/urbansound8k_waveform.py`、`configs/pretrained_cnn_*.yaml`、`scripts/run_pretrained_cnn_transfer.py`、`docs/experiments/pretrained-cnn-transfer.md` | 待使用者確認 Colab development 結果與是否進入 partial fine-tuning | Fold 10 封存；原始資料與既有 Mel cache不修改；第三方來源與 MIT 授權已保留 |
| 2026-08-17 | 操作 Colab A100 完成 EfficientAT linear probing、partial fine-tuning 與唯一鄰近設定，以 folds 1、4、7 mean validation Macro F1 鎖定模型後執行一次 fold 10 final evaluation，並核對 Drive artifacts | `docs/experiments/pretrained-cnn-transfer.md`、`configs/pretrained_cnn_partial_finetune_lr2e5.yaml`、`configs/pretrained_cnn_partial_finetune_final_test.yaml`；Drive transfer artifacts | 使用者需確認是否投入正式 10-fold | Development F1 `0.8716 ± 0.0283`；fold 10 test F1 `0.9041`；test 未參與選模 |
| 2026-08-17 | 實作並操作 Colab A100 完成 EfficientAT v2 development-only study，受控比較 8 epochs、gradual unfreezing、輕量 masking、weak Mixup 與 last 1/2/3 blocks | `src/train_pretrained_cnn.py`、`src/models/pretrained_efficientat.py`、`configs/pretrained_cnn_v2_*.yaml`、`docs/experiments/pretrained-cnn-transfer-v2.md`；Drive v2 artifacts | 使用者需確認是否以唯一勝出方法執行正式 10-fold | 勝出 development F1 `0.8844 ± 0.0165`；所有 v2 runs 均未執行 test；未修改 raw data 或 cache |
| 2026-08-17 | 設計並實作 EfficientAT recommended study，加入動態 waveform augmentation、zero-fill time-shift TTA、固定三 seed probability ensemble、MN20 transfer learning 及鎖定式正式 10-fold runner | `src/data/waveform_augmentation.py`、`src/pretrained_cnn_inference.py`、`src/evaluate_pretrained_cnn.py`、`scripts/run_pretrained_cnn_*`、`configs/pretrained_cnn_recommended_*.yaml`、`docs/experiments/pretrained-cnn-recommended-study.md` | 待 Colab development 結果完成後依 validation Macro F1 鎖定唯一方法 | Development 前不讀 fold 10；不覆寫 raw audio 或 cache；大型 artifacts 只備份到 Drive |
| 2026-08-17 | 操作 Colab A100 完成 recommended development study，受控比較三種 waveform augmentation、time-shift TTA、固定 MN10 三 seed ensemble、MN20 linear probe 與兩個 partial-unfreezing depths，並只依 folds 1、4、7 mean validation Macro F1 鎖定 MN20 last-2 | `configs/pretrained_cnn_mn20_locked_last2.yaml`、`docs/experiments/pretrained-cnn-recommended-study.md`；Drive `pretrained_cnn_recommended/` | 執行唯一固定方法的 formal 10-fold | 勝出 F1 `0.89069 ± 0.01165`；鎖定時未評估 test；沒有覆寫 raw audio 或 cache |
| 2026-08-17 | 在鎖定 commit `78a3245` 後操作 Colab A100 執行唯一 MN20 last-2 formal 10-fold；每個 test fold 只評估一次，核對 10 組 predictions/checkpoints、aggregate summary 與 Drive backup | `scripts/run_pretrained_cnn_cross_validation.py`、`docs/experiments/pretrained-cnn-recommended-study.md`；Drive `pretrained_cnn_recommended/pretrained_cnn_mn20_locked_last2_formal_10fold_1seed/` | 將正式 mean/std 與 per-class F1 納入論文；不得再依 test folds 調參 | Formal F1 `0.87686 ± 0.04048`、Accuracy `0.86883 ± 0.04263`；seed 42、無 TTA；raw/cache 未修改 |
| 2026-08-17 | 設計 MN20 post-formal development-only 探索，實作 class-balanced focal loss、validation top-k checkpoint weight averaging、共享 linear-probe 初始化與固定三 seed probability evaluation | `src/losses.py`、`src/checkpoint_averaging.py`、`scripts/run_pretrained_cnn_postformal_study.py`、`configs/pretrained_cnn_mn20_postformal_*.yaml`、`docs/experiments/pretrained-cnn-postformal-exploration.md` | 待 Colab folds 1/4/7 validation 結果 | 正式 test 結果不得參與此研究選擇；不修改 raw data 或 cache |

## 可記錄的協助類型

- 論文架構與章節草稿。
- 段落潤飾與語氣調整。
- 文獻摘要與比較表。
- 程式碼撰寫、除錯與重構。
- 實驗設計、結果整理與圖表產生。
- 研究日誌、會議紀錄與任務管理。
- 可重複研究與交付前檢查。

## 定期檢查

每週整理進度時，請確認：

- [ ] 本週是否使用 AI 協助產生重要內容？
- [ ] 是否已記錄到本文件？
- [ ] 是否有內容需要使用者或教授確認？
- [ ] 是否有學校規範要求特定揭露方式？
