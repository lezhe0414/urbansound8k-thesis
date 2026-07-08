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
