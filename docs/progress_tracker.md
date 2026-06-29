# MVP 進度追蹤

更新日期：2026-06-29

這份文件用來集中追蹤 UrbanSound8K 聲音分類 MVP 的已完成、未完成與下一步。之後每次程式、資料、實驗、圖表或論文草稿有重要更新時，都要同步更新本文件。

## 目前結論

MVP 已完成，可展示端到端流程：UrbanSound8K 音訊已下載驗證，已轉成 Mel-spectrogram，CNN baseline 與 Spectrogram Transformer 都能訓練、評估並輸出 metrics 與 confusion matrix。Spectrogram Transformer fold 10 已完成正式 10 epochs 訓練。

整個論文專案尚未完成，因為 CNN baseline 正式長訓練、結果解讀、文獻整理與 8 頁論文 PDF 仍待補。

## 已做到

| 項目 | 狀態 | 證據 / 路徑 | 備註 |
| --- | --- | --- | --- |
| 建立 Python 專案依賴 | 完成 | `requirements.txt` | 包含 PyTorch、Librosa、soundata 等 |
| 下載 UrbanSound8K | 完成 | `data/raw/UrbanSound8K_soundata/` | 8732 個音訊檔，不提交到 git |
| 驗證 UrbanSound8K | 完成 | `soundata.validate()` | 已確認資料集完整 |
| 音訊轉 Mel-spectrogram | 完成 | `data/processed/urbansound8k_mels/` | 8732 筆處理後資料，不提交到 git |
| 資料集 loader | 完成 | `src/data/urbansound8k.py` | 支援 fold split、preload、smoke sample limit |
| CNN baseline 模型 | 完成 | `src/models/cnn.py` | 架構可訓練，正式長訓練待補 |
| Spectrogram Transformer 模型 | 完成 | `src/models/spectrogram_transformer.py` | 作為現代比較模型 |
| Preprocessing script | 完成 | `src/preprocess.py` | 可重複產生 Mel-spectrogram |
| Training script | 完成 | `src/train.py` | 可輸出 checkpoint、history、metrics、confusion matrix |
| Evaluation script | 完成 | `src/evaluate.py` | 可重讀 checkpoint 重新評估 |
| CNN smoke run | 完成 | `results/cnn_baseline_smoke_fold10/` | 只驗證流程，非正式結果 |
| Transformer smoke run | 完成 | `results/transformer_baseline_smoke_fold10/` | 只驗證流程，非正式結果 |
| Transformer fold 10 正式訓練 | 完成 | `results/transformer_baseline_fold10/` | 10 epochs，完整 train/val/test split |
| Transformer confusion matrix | 完成 | `figures/transformer_baseline_fold10_confusion_matrix.png` | 可用於初步討論 |
| 單元測試 | 完成 | `tests/` | `python -m unittest discover -s tests` 通過 |
| 專案狀態檢查 | 完成 | `scripts/check_project_status.py` | 可快速檢查文件與下一步 |
| Git commit | 完成 | `acda41d`、`7325b43` | 程式與狀態文件已提交 |

## 已取得結果

### Transformer fold 10 正式結果

| 指標 | 數值 |
| --- | ---: |
| Accuracy | 0.6547 |
| Macro precision | 0.6879 |
| Macro recall | 0.6711 |
| Macro F1 | 0.6644 |
| Test loss | 1.1418 |

結果來源：`results/transformer_baseline_fold10/metrics.json`

### Smoke run 說明

Smoke run 只用少量資料與 1 epoch 檢查 pipeline 是否能完整執行。它不是正式實驗結果，不應直接寫成論文主要分數。

目前 smoke 設定：

| 設定 | 數值 |
| --- | ---: |
| Epochs | 1 |
| Train samples | 256 |
| Validation samples | 128 |
| Test samples | 128 |
| Batch size | 32 |
| Train batches per epoch | 8 |
| Validation batches per epoch | 4 |
| Test batches | 4 |

## 尚未做到

| 項目 | 狀態 | 原因 / 風險 | 下一步 |
| --- | --- | --- | --- |
| CNN baseline 正式長訓練 | 未完成 | 本機 CPU 上 CNN 訓練偏慢 | 建議用 Colab/GPU 或調整模型/epochs 後再跑 |
| 10-fold cross validation | 未完成 | MVP 先跑 fold 10 | 後續可跑 `--fold all` 或多 fold 平均 |
| CNN vs Transformer 正式比較表 | 未完成 | CNN 正式分數尚缺 | 補 CNN 正式訓練後整理表格 |
| 結果圖表解讀 | 未完成 | 目前只有 confusion matrix 與 metrics | 寫出哪些類別容易混淆、可能原因 |
| 文獻整理 | 未完成 | 還沒整理核心 citation | 補 UrbanSound8K、CNN spectrogram、Transformer/AST 相關文獻 |
| 方法章草稿 | 未完成 | 需要把 pipeline 寫成論文語言 | 先寫 Mel-spectrogram + model comparison 方法 |
| 結果與討論草稿 | 未完成 | CNN 正式結果缺失 | 先寫 Transformer 初步結果，CNN 後補 |
| 8 頁 PDF | 未完成 | 需要正文、圖表、引用與排版 | 在結果與方法穩定後生成 |
| 教授確認 Transformer 策略 | 未完成 | 需確認是否接受偏離原 CNN-only definition | 週五會議時說明 CNN baseline + Transformer comparison |

## 下一步優先順序

1. 準備教授討論材料：說明目前 MVP、Transformer fold 10 結果、CNN 正式長訓練尚缺。
2. 決定 CNN 正式訓練環境：本機繼續跑、縮小設定，或改用 Colab/GPU。
3. 補 CNN baseline 正式結果，形成 CNN vs Transformer 對照。
4. 整理 confusion matrix 與 metrics 成論文可用圖表與表格。
5. 開始撰寫方法章與初步結果段落。

## 常用命令

安裝依賴：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

重新 preprocessing：

```bash
python3 -m src.preprocess \
  --raw-dir data/raw/UrbanSound8K_soundata \
  --out-dir data/processed/urbansound8k_mels
```

快速 smoke run：

```bash
python3 -m src.train --config configs/cnn_smoke.yaml --fold 10
python3 -m src.train --config configs/transformer_smoke.yaml --fold 10
```

正式 Transformer fold 10：

```bash
python3 -m src.train --config configs/transformer_baseline.yaml --fold 10
python3 -m src.evaluate --run-dir results/transformer_baseline_fold10
```

正式 CNN fold 10：

```bash
python3 -m src.train --config configs/cnn_baseline.yaml --fold 10
python3 -m src.evaluate --run-dir results/cnn_baseline_fold10
```

測試與狀態檢查：

```bash
python3 -m unittest discover -s tests
python3 scripts/check_project_status.py
```

## 維護規則

- 每次新增或修改程式、設定、資料處理流程、實驗結果、圖表或論文草稿時，都同步更新本文件。
- 若結果只是 smoke run，必須標示為非正式結果。
- 若結果可放進論文，必須記錄 metrics、路徑、資料 split 與模型設定。
- 大型資料、processed features、results、figures 目前不提交到 git，但路徑要記錄在本文件。
