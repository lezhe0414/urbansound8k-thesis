# 專案目前狀態

更新日期：2026-08-17

## 專案目的

本專案用來協助完成論文寫作、程式碼開發、資料整理、實驗紀錄、圖表產出與教授回饋追蹤。

教授已同意使用 AI Agent，因此本專案已把 AI Agent 協作流程納入正式工作方式。

## 已確認的 project definition

- 題目：Sound Event Detection Using Machine Learning Techniques
- 學校：Queen Mary University of London
- 學生：CHE LI
- 指導教授：Lin Wang
- 核心任務：建立 deep learning-based sound event detection system。
- 技術路線：音訊資料轉成 spectrogram/Mel-spectrogram，再以 CNN 分類聲音事件。
- 範例類別：bird sounds、drone sounds、background noise。
- 可用資料集：UrbanSound8K 或 ESC-50。
- 評估指標：accuracy、precision、recall、F1-score，可加 confusion matrix。
- 工具：Python、NumPy、PyTorch、Librosa、Matplotlib。
- 原 timeplan：6 月初文獻與資料處理，6 月底 baseline CNN 初步結果，7 月 9 日前 draft dissertation，8 月 19 日前 final submission。

## 目前判斷

目前不是完成狀態。UrbanSound8K 的端到端 pipeline、CNN、從零訓練 Spectrogram Transformer、即時 spectrogram augmentation 與受控搜尋流程均已完成。CNN 受控搜尋只依 validation Macro F1 選模，鎖定唯一設定後才執行一次 fold 10 test。

目前最重要的實驗進度如下：

1. 已下載 UrbanSound8K 到 `data/raw/UrbanSound8K_soundata/`，共 8732 個音訊檔，並已通過 `soundata.validate()`。
2. 已執行 Mel-spectrogram preprocessing，輸出到 `data/processed/urbansound8k_mels/`。
3. CNN 受控 augmentation 搜尋完成：最佳 validation Macro F1 `0.7924`、validation Accuracy `0.7709`。
4. 唯一最佳設定的單次 fold 10 test：Accuracy `0.8471`、Macro F1 `0.8536`。
5. 歷史 CNN fold 10 Macro F1 約 `0.8413`；本次差異約 `+0.0123`，尚需 10-fold 統計驗證。
6. 從零訓練 Spectrogram Transformer 表現低於 CNN，因此保留為比較模型，不再用 fold 10 反覆調參。
7. EMA validation-only 比較已完成：EMA Macro F1 只比同次 online 權重高約 `0.00089`，不足以支持採用，正式設定關閉 EMA。
8. 固定 3-seed probability ensemble 已完成：validation Macro F1 `0.7699`，低於鎖定單一 CNN 的 `0.7924`；因此完成但不採用。
9. 鎖定後唯一一次 ensemble fold 10 test Accuracy 為 `0.8411`、Macro F1 為 `0.8501`，亦略低於單一 CNN，但此 test 只作確認，不作選模依據。
10. EfficientAT `mn10_as` transfer-learning study 已完成：linear probing development Macro F1 `0.8471 ± 0.0327`；partial fine-tuning 最佳設定為 `0.8716 ± 0.0283`，比相同 protocol 的 control `0.7818` 高 `0.0898`。
11. 唯一設定鎖定後只執行一次 fold 10 test，Accuracy `0.8949`、Macro F1 `0.9041`；test 不參與選模或後續調參。
12. EfficientAT v2 development-only study 已完成：8 epochs、weak Mixup (`alpha=0.15`, `p=0.5`) 與解凍最後 3 個 blocks 的唯一勝出設定達 Macro F1 `0.8844 ± 0.0165`、Accuracy `0.8824 ± 0.0141`。V2 沒有再次執行 fold 10；上述 v1 test 分數不能當作 v2 test result。

## 已建立內容

### 協作規範

- `AGENTS.md`：AI Agent 協作規則。
- `README.md`：專案入口與工作流程。
- `docs/dashboard.md`：專案儀表板。
- `docs/ai_workflow.md`：如何讓 AI Agent 協助寫作、文獻、程式與實驗。
- `docs/ai_usage_log.md`：AI Agent 使用紀錄。
- `docs/ai_disclosure_draft.md`：AI 使用揭露草稿。

### 論文規劃

- `docs/intake_questions.md`：論文需求訪談表。
- `docs/next_input_template.md`：下一次給 AI Agent 的填空範本。
- `docs/first_week_plan.md`：第一週啟動計畫。
- `docs/professor_questions.md`：要向教授確認的問題。
- `docs/professor_update_template.md`：給教授的更新與詢問模板。
- `docs/thesis_plan.md`：論文計畫。
- `docs/thesis_outline.md`：論文章節大綱。
- `docs/chapters/`：逐章草稿。
- `docs/writing_style_guide.md`：論文寫作風格指南。
- `docs/glossary.md`：術語表。

### 進度與決策管理

- `docs/milestones.md`：里程碑與任務追蹤。
- `docs/weekly_review.md`：每週進度檢查。
- `docs/task_inbox.md`：未分類任務入口。
- `docs/decision_log.md`：重要決策紀錄。
- `docs/artifact_index.md`：成果索引。
- `docs/risk_register.md`：風險與阻塞追蹤。
- `docs/research_log.md`：研究日誌。
- `docs/meeting_notes.md`：教授會議紀錄。

### 程式與實驗

- `scripts/check_project_status.py`：專案狀態檢查腳本。
- `requirements.txt`：Python 依賴，包含 PyTorch、Librosa、soundata 等。
- `src/preprocess.py`：UrbanSound8K 音訊轉 Mel-spectrogram。
- `src/train.py`：訓練、驗證、測試、checkpoint、metrics 與 confusion matrix 輸出。
- `src/evaluate.py`：重讀 checkpoint 並產生評估結果。
- `configs/cnn_baseline.yaml`、`configs/transformer_baseline.yaml`：正式訓練設定。
- `configs/cnn_smoke.yaml`、`configs/transformer_smoke.yaml`：本機快速 smoke run 設定。
- `src/README.md`：正式程式碼放置規則。
- `notebooks/README.md`：探索性 notebook 規則。
- `docs/environment.md`：程式環境與執行方式。
- `docs/code_task_spec.md`：程式任務規格模板。
- `.env.example`：環境變數樣板。
- `docs/experiment_template.md`：實驗紀錄模板。
- `docs/reproducibility_checklist.md`：可重複研究檢查清單。
- `docs/submission_checklist.md`：交付前檢查清單。

### 資料、結果與圖表

- `data/README.md`：資料管理規則。
- `data/raw/README.md`：原始資料規則。
- `data/processed/README.md`：處理後資料規則。
- `results/README.md`：實驗結果管理規則。
- `figures/README.md`：圖表管理規則。

### 文獻與引用

- `references/README.md`：文獻與引用管理流程。
- `references/literature_notes.md`：文獻筆記。
- `references/citation_tracker.md`：引用追蹤表。
- `references/references.bib`：BibTeX 參考文獻檔。

## 尚未取得或尚待確認的必要資訊

目前已能開始寫程式，但仍需確認：

1. 本機是否要訓練，或改用 Google Colab/GPU。
2. 教授是否接受 Transformer 作為主要比較模型。
3. 學校正式引用格式與 8 頁 PDF 排版要求。
4. 7 月 9 日 draft 是否仍需提交，以及目前教授是否有新的優先事項。

## 下一步

第三模型的 v2 validation-only 方法選擇已完成。下一步是固定設定的正式統計驗證：

1. 以 `configs/cnn_aug_final.yaml` 執行固定 from-scratch CNN 10-fold，不再依 fold 結果調參。
2. 時間允許時，以 `configs/pretrained_cnn_v2_mixup_last3.yaml` 的已鎖定方法參數建立 EfficientAT fixed-config 10-fold runner；每個 test fold 必須完全排除於選模，並只使用其餘 folds 建立 validation split。
3. 整理 from-scratch CNN、from-scratch Transformer 與 AudioSet-pretrained CNN 的公平比較，明確區分從零訓練與 AudioSet transfer learning。
4. 將 10-fold mean/std、per-class F1 與 confusion matrix 納入論文結果及限制分析。

## 目前完成度判斷

專案基礎設施、資料處理、模型 pipeline、CNN 受控 augmentation 搜尋、EMA、3-seed ensemble，以及 EfficientAT v1/v2 transfer-learning studies 均已完成。V2 僅使用 development validation 並未再次查看 fold 10。整體目標尚未完成，因為固定設定的正式 10-fold 統計、最終圖表解讀與 8 頁論文仍需完成。
