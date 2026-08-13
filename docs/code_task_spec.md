# 程式任務規格：Sound Event Detection Baseline

這份文件定義目前最優先的程式任務：建立 sound event detection 的最小可交付 pipeline。目標是先符合 project definition 的 CNN baseline，再視時間加入 Transformer 比較。

## 使用時機

- 要建立資料處理腳本前。
- 要建立模型、演算法或分析方法前。
- 要建立系統原型、API、前端或工具前。
- 要產生論文圖表或表格前。
- 要重構 notebook 成正式程式前。

## 任務規格

### 任務名稱

Audio spectrogram preprocessing + CNN baseline classification

### 任務狀態

- 狀態：待開發
- 建立日期：2026-06-28
- 更新日期：2026-06-28
- 負責人：CHE LI + AI Agent

### 對應論文內容

- 對應研究問題：CNN-based sound event detection on spectrogram representations
- 對應論文章節：methodology、experiments、results and discussion
- 對應教授要求：project definition 指定 CNN models for sound event classification
- 是否支撐某個論文主張：支撐「spectrogram-based CNN can classify sound events」與「model configurations can be compared」

### 任務目的

這個程式要解決什麼問題？

- 將公開音訊資料集轉換為可訓練的 spectrogram tensors。
- 訓練 baseline CNN 進行聲音事件分類。
- 輸出可放入論文的 metrics 與圖表。

### 輸入

| 輸入 | 來源 | 格式 | 是否必填 | 備註 |
| --- | --- | --- | --- | --- |
| Audio files | UrbanSound8K 或 ESC-50 | `.wav` | 是 | 先支援一個資料集 |
| Labels / metadata | Dataset metadata | `.csv` 或資料夾類別結構 | 是 | 需映射到 class index |
| Config | 本專案 | `.yaml` 或 CLI args | 否 | 取樣率、spectrogram 參數、訓練參數 |

### 輸出

| 輸出 | 位置 | 格式 | 用途 | 對應章節 |
| --- | --- | --- | --- | --- |
| Processed dataset | `data/processed/` | tensors 或 cached arrays | 加速訓練與重現 | Methodology |
| Model checkpoint | `results/models/` | `.pt` | 保存訓練模型 | Experiments |
| Metrics | `results/metrics/` | `.json` / `.csv` | 報告模型表現 | Results |
| Confusion matrix | `figures/` | `.png` | 論文圖表 | Results |

### 方法或邏輯

請描述核心方法、流程或演算法。

1. 讀取音訊資料與 labels。
2. 統一取樣率與固定音訊長度，不足補零，過長裁切。
3. 產生 Mel-spectrogram，轉成 log scale 並 normalise。
4. 建立 train/validation/test split。
5. 訓練 CNN baseline。
6. 在 validation/test set 上計算 accuracy、precision、recall、F1-score。
7. 輸出 confusion matrix 與可追蹤的實驗設定。
8. 若 baseline 完成，再加入 Transformer 或 transfer learning model 作比較。

### 執行方式

```text
python3 -m src.preprocess --raw-dir data/raw/UrbanSound8K --out-dir data/processed/urbansound8k_mels
python3 -m src.train --config configs/cnn_baseline.yaml --fold 10
python3 -m src.train --config configs/transformer_baseline.yaml --fold 10
python3 -m src.evaluate --run-dir results/transformer_baseline_fold10
```

### 驗證方式

這個程式如何確認是正確或可用的？

- [ ] 可成功執行。
- [ ] 有明確輸入與輸出。
- [ ] 產出結果符合預期格式。
- [ ] 結果可對應論文章節或實驗紀錄。
- [ ] 錯誤情境有處理或記錄。
- [ ] 至少能用一小批資料跑完整 pipeline。
- [ ] metrics 與 confusion matrix 可被論文引用。

### 實驗或結果紀錄

- 對應實驗紀錄：`docs/experiment_template.md` 的第一份實驗紀錄
- 對應結果檔：`results/metrics/`
- 對應圖表：`figures/`
- 對應 artifact index 項目：CNN baseline、processed dataset、evaluation results

### 依賴與環境

- 程式語言：Python
- 套件 / 框架：PyTorch、Librosa、NumPy、scikit-learn、Matplotlib
- 外部服務：可選 Google Colab
- 環境變數：`DATA_DIR`、`RESULTS_DIR`、`FIGURES_DIR`
- 硬體需求：CPU 可跑小樣本；完整訓練建議 GPU 或 Colab

### 風險與待確認

- 資料集尚未下載，需確認使用 UrbanSound8K 或 ESC-50。
- Transformer 不應先取代 CNN；需要教授確認或放在 extension。

## 完成標準

- [ ] 規格已填完整。
- [ ] 程式碼已放在 `src/` 或 notebook 已放在 `notebooks/`。
- [ ] 執行方式已寫入 `docs/environment.md` 或相關 README。
- [ ] 結果已保存到 `results/` 或 `figures/`。
- [ ] 實驗已用 `docs/experiment_template.md` 或同等格式記錄。
- [ ] 成果已更新到 `docs/artifact_index.md`。
- [ ] 若支撐論文內容，已更新對應章節。

---

## 程式任務規格：CNN Spectrogram Data Augmentation Ablation

- 狀態：程式與 Colab 正式實驗均完成；受控搜尋結果已鎖定
- 更新日期：2026-08-13
- 目的：在不更改 CNN 架構與 preprocessing 的條件下，量化不同資料增強強度對 validation Macro F1 的影響。
- 輸入：已快取的 normalized Mel-spectrogram、UrbanSound8K 官方 fold、augmentation YAML config。
- 方法：逐樣本時間位移、頻率位移、時間伸縮、強度縮放、Gaussian noise、time/frequency masking，以及逐樣本 Mixup 或 spectrogram CutMix。
- 控制：control、light、balanced、strong 四組使用相同 seed、模型、optimizer、sampling 與 epoch，只改 augmentation。
- 模型選擇：只依 validation Macro F1 選擇設定；fold 10 test 不用於選參數，僅評估勝出設定一次。
- 輸出：各組設定檔、history、validation metrics、checkpoint、training history 圖、逐輪 CSV/Markdown 紀錄，以及唯一勝出設定的 test metrics 與 confusion matrix。

執行方式：

```text
python3 scripts/run_cnn_augmentation_ablation.py --fold 10 --skip-existing
```

正式受控流程使用 `scripts/run_cnn_controlled_search.py`。初始四組完成後只延伸當前最佳設定；每輪只改一類變因，未改善即退回，並以 validation Macro F1 連續五輪未改善作為提早停止條件。Test evaluation 預設關閉，必須明確加入 `--final-test`，且只會在唯一設定鎖定後執行一次。

驗證證據：

- `tests/test_augmentation.py` 驗證形狀、有限值、零填充位移、Mixup/CutMix、舊設定相容性與參數錯誤處理。
- `tests/test_controlled_search.py` 驗證初始四組控制條件一致、選模只依 validation Macro F1，以及所有預定迭代可產生有效變更。
- `configs/cnn_aug_smoke.yaml` 已用真實 processed data 完成 1 epoch end-to-end smoke run。

---

## 程式任務規格：CNN 3-seed Probability Ensemble

- 狀態：程式與 Colab 正式實驗均完成；結果不支持採用 ensemble
- 更新日期：2026-08-13
- 基礎設定：`configs/cnn_aug_final.yaml`
- Seeds：42、123、2026
- EMA：明確關閉
- 個別模型選擇：各 seed 只依 validation Macro F1 保存最佳 checkpoint。
- Ensemble：平均三個模型的 softmax probability，不選擇單一最佳 seed。
- Test 規範：個別 seed 禁止 test evaluation；`--run-test` 只允許對已鎖定的 ensemble 執行一次，若 metrics 已存在則拒絕重跑。
- 輸出：每個 seed 的 validation run、seed summary、ensemble validation/test metrics、預測機率與 confusion matrix。

執行方式：

```text
python3 -m src.ensemble --config configs/cnn_aug_final.yaml --fold 10 --seeds 42 123 2026
python3 -m src.ensemble --config configs/cnn_aug_final.yaml --fold 10 --seeds 42 123 2026 --skip-existing --run-test
```

驗證證據：`tests/test_seed_ensemble.py` 檢查 EMA 與個別 test 強制關閉、三個 seed 必須互異，以及 probability arithmetic mean 的正確性。

正式結果：ensemble validation Macro F1 為 `0.7699`，低於受控搜尋鎖定單一 CNN 的 `0.7924`；唯一一次 fold 10 test Macro F1 為 `0.8501`，亦略低於單一 CNN 的 `0.8536`。完整紀錄見 `docs/experiments/2026-08-13-cnn-seed-ensemble.md`。
