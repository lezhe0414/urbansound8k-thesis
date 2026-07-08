# 程式環境設定

這份文件記錄 sound event detection 程式碼需要的執行環境。實際版本會在建立程式碼與依賴檔後補齊。

## 基本資訊

- 主要程式語言：Python
- 作業系統：macOS / Google Colab
- 套件管理工具：pip 或 conda
- 主要框架：PyTorch、Librosa、NumPy、scikit-learn、Matplotlib
- 資料庫或外部服務：無；可選 Google Colab
- GPU / CPU 需求：CPU 可跑小樣本；完整訓練建議 GPU

## 安裝方式

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 執行方式

```bash
python3 -m src.preprocess --raw-dir data/raw/UrbanSound8K --out-dir data/processed/urbansound8k_mels
python3 -m src.train --config configs/cnn_baseline.yaml --fold 10
python3 -m src.train --config configs/transformer_baseline.yaml --fold 10
python3 -m src.evaluate --run-dir results/transformer_baseline_fold10
```

正式 10-fold cross validation：

```bash
python3 -m src.train --config configs/cnn_baseline.yaml --fold all
python3 -m src.train --config configs/transformer_baseline.yaml --fold all
```

每個模型會輸出：

```text
results/<run_name>_fold1/ ... results/<run_name>_fold10/
results/<run_name>_10fold_summary.json
results/<run_name>_10fold_summary.csv
```

## Google Colab CNN baseline

若本機 CPU 訓練 CNN 太慢，可使用下列 notebook 在 Colab GPU 上執行正式 CNN fold 10：

```text
notebooks/2026-07-02-colab-cnn-baseline.ipynb
```

此流程以 GitHub repo 作為程式碼來源。Colab runtime 只負責重新下載 UrbanSound8K、產生 Mel-spectrogram cache、訓練 CNN、評估並打包輸出。跑完後需將 `results/cnn_baseline_fold10/` 和對應 `figures/` 圖檔下載回本地。

若要在 Colab 同時跑 CNN baseline 與 Spectrogram Transformer fold 10，使用：

```text
notebooks/2026-07-08-colab-cnn-transformer-fold10.ipynb
```

這個 notebook 使用 Google Drive 保存 UrbanSound8K raw audio 與 processed Mel-spectrogram cache，避免每次 Colab runtime 重開都重新下載或 preprocessing。

沒有正式資料時，可先建立 synthetic dataset 檢查 pipeline：

```bash
python3 scripts/create_synthetic_urbansound8k.py --out-dir data/raw/UrbanSound8K_synthetic
python3 -m src.preprocess --raw-dir data/raw/UrbanSound8K_synthetic --out-dir data/processed/urbansound8k_synthetic_mels
```

## 環境變數

實際環境變數放在 `.env`，不要提交到 Git。可公開的樣板放在 `.env.example`。

| 變數名稱 | 用途 | 是否必填 | 範例 |
| --- | --- | --- | --- |
| `DATA_DIR` | 資料根目錄 | 否 | `data` |
| `RESULTS_DIR` | 結果輸出目錄 | 否 | `results` |
| `FIGURES_DIR` | 圖表輸出目錄 | 否 | `figures` |

## 版本紀錄

| 工具 | 版本 | 備註 |
| --- | --- | --- |
| Python | 3.10+ 建議 | 使用本機或 Colab 實際環境確認 |
| PyTorch | 見 `requirements.txt` | 模型訓練 |
| Librosa | 見 `requirements.txt` | 音訊處理與 Mel-spectrogram |

## 重現步驟

當程式流程確定後，應補上從原始資料到論文圖表的完整流程：

1. 取得或放置原始資料。
2. 執行 `src.preprocess` 產生 Mel-spectrogram cache。
3. 執行 `src.train` 訓練 CNN baseline 與 Spectrogram Transformer。
4. 執行 `src.evaluate` 產生 metrics。
5. 從 `figures/` 取得 confusion matrix。
6. 將結果寫入論文章節。

## 注意事項

- 不要把 API key、密碼、私人路徑寫進 Git。
- 若使用大型模型或大型資料，需記錄硬體需求與執行時間。
- 若環境設定會影響實驗結果，需在實驗紀錄中註明。
