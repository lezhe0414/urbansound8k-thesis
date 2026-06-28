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
python3 -m src.preprocess --dataset urban_sound_8k --raw-dir data/raw --out-dir data/processed
python3 -m src.train --config configs/cnn_baseline.yaml
python3 -m src.evaluate --checkpoint results/models/cnn_baseline.pt --split test
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
