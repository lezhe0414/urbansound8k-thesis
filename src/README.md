# 程式碼目錄

`src/` 用來放正式、可重複執行的程式碼。探索中的程式可以先放在 `notebooks/`，確認需要重複執行後再整理到這裡。

## 建議結構

目前程式主線已確定為 UrbanSound8K spectrogram sound classification：

```text
src/
├── README.md
├── preprocess.py              # UrbanSound8K audio -> Mel-spectrogram cache
├── train.py                   # CNN / Transformer training
├── evaluate.py                # trained run evaluation
├── data/urbansound8k.py       # processed dataset loader
├── models/cnn.py              # CNN baseline
├── models/spectrogram_transformer.py
└── utils/                     # config, metrics, plotting, seed helpers
```

CNN 是 baseline；Spectrogram Transformer 是主要比較模型。

## 程式碼最低要求

寫正式程式前，先用 `docs/code_task_spec.md` 定義任務規格。每個正式腳本應該說明：

- 目的：這段程式解決什麼論文問題？
- 輸入：讀取哪些資料或參數？
- 輸出：產生哪些結果、表格或圖？
- 執行方式：用什麼命令可以重複執行？
- 對應章節：支撐論文哪一章或哪個研究問題？
- 驗證方式：如何確認輸出正確且可重現？

## 命名建議

- 資料處理：`prepare_data.*`、`clean_data.*`
- 實驗執行：`run_experiment.*`
- 評估：`evaluate.*`
- 圖表：`make_figures.*`
- 共用工具：`utils.*`

## 目前可執行命令

```bash
python3 -m src.preprocess --raw-dir data/raw/UrbanSound8K --out-dir data/processed/urbansound8k_mels
python3 -m src.train --config configs/cnn_baseline.yaml --fold 10
python3 -m src.train --config configs/transformer_baseline.yaml --fold 10
python3 -m src.evaluate --run-dir results/transformer_baseline_fold10
```

## 從 notebook 移到 src 的標準

符合以下條件時，應該把 notebook 裡的程式整理到 `src/`：

- 需要重複執行；
- 會影響論文結果；
- 教授或口試委員可能要求說明；
- 需要讓其他人重現；
- 已經不是單純探索。
