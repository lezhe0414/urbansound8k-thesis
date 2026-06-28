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

```text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchaudio librosa numpy scikit-learn matplotlib pandas tqdm pyyaml
```

## 執行方式

```text
python3 -m src.preprocess --dataset urban_sound_8k --raw-dir data/raw --out-dir data/processed
python3 -m src.train --config configs/cnn_baseline.yaml
python3 -m src.evaluate --checkpoint results/models/cnn_baseline.pt --split test
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
| Python | 待確認 | 使用本機或 Colab 實際環境確認 |
| PyTorch | 待確認 | 模型訓練 |
| Librosa | 待確認 | 音訊處理與 Mel-spectrogram |

## 重現步驟

當程式流程確定後，應補上從原始資料到論文圖表的完整流程：

1. 取得或放置原始資料。
2. 執行資料處理。
3. 執行模型、演算法或分析。
4. 執行評估。
5. 產生圖表。
6. 將結果寫入論文章節。

## 注意事項

- 不要把 API key、密碼、私人路徑寫進 Git。
- 若使用大型模型或大型資料，需記錄硬體需求與執行時間。
- 若環境設定會影響實驗結果，需在實驗紀錄中註明。
