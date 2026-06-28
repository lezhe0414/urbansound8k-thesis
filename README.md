# 論文專案

這個專案用來協助完成論文寫作、程式碼開發、實驗紀錄、資料整理與結果產出。

教授已同意使用 AI Agent，因此本專案會把 AI 協作流程納入正式工作方式：每次修改都盡量留下可追蹤的文件、程式、實驗紀錄或結果。

## 專案結構

```text
.
├── AGENTS.md
├── README.md
├── docs/
├── src/
├── notebooks/
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── figures/
└── references/
```

## 主要資料夾

- `docs/`：論文計畫、章節大綱、章節草稿、研究日誌、會議紀錄。
- `src/`：正式可重複執行的程式碼。
- `notebooks/`：探索性分析或實驗筆記。
- `data/raw/`：原始資料，原則上不覆蓋、不直接修改。
- `data/processed/`：清理後或轉換後的資料。
- `results/`：實驗輸出、指標、表格與紀錄。
- `figures/`：論文用圖、流程圖、結果圖。
- `references/`：文獻、BibTeX、閱讀筆記。

## 建議工作流程

1. 先看 `docs/dashboard.md`，快速確認目前階段、最高風險與下一步。
2. 再看 `docs/current_status.md`，確認完整完成度與資訊缺口。
3. 依 `docs/first_week_plan.md` 補齊第一週啟動資訊。
4. 在 `docs/thesis_plan.md` 補上題目、研究問題、方法與目前限制。
5. 如果還不確定怎麼描述需求，先看 `docs/next_input_template.md` 或填 `docs/intake_questions.md`。
6. 不確定要問教授什麼時，先看 `docs/professor_questions.md`。
7. 要傳訊息給教授時，使用 `docs/professor_update_template.md`。
8. 不確定要放哪裡的事項，先放進 `docs/task_inbox.md`。
9. 重要方向、技術選型或教授要求改變時，記到 `docs/decision_log.md`。
10. 用 `docs/artifact_index.md` 追蹤所有重要成果。
11. 用 `docs/risk_register.md` 追蹤風險與阻塞。
12. 用 `docs/milestones.md` 追蹤論文、程式、實驗與教授回饋。
13. 每次與教授討論後，把結論寫進 `docs/research_log.md` 與 `docs/meeting_notes.md`。
14. 重要 AI 協助記錄到 `docs/ai_usage_log.md`。
15. 寫作時依 `docs/writing_style_guide.md` 與 `docs/glossary.md` 統一語氣和術語。
16. 每週用 `docs/weekly_review.md` 檢查進度與風險。
17. 可執行 `python3 scripts/check_project_status.py` 快速檢查專案狀態。
18. 章節正式草稿放在 `docs/chapters/`。
19. 寫程式前先用 `docs/code_task_spec.md` 定義任務規格。
20. 程式原型可先放 `notebooks/`，確定要重複執行後整理到 `src/`。
21. 程式環境與執行方式記在 `docs/environment.md`。
22. 每次實驗前後參考 `docs/experiment_template.md` 紀錄目的、命令、參數與結果。
23. 用 `docs/reproducibility_checklist.md` 檢查程式、資料與實驗是否能支撐論文。
24. 交給教授或階段交付前，用 `docs/submission_checklist.md` 檢查。
25. 每次實驗輸出放到 `results/`，圖表放到 `figures/`。
26. 每篇重要文獻都在 `references/literature_notes.md` 留下摘要、方法、可引用觀點與疑問，引用需求追蹤在 `references/citation_tracker.md`。

## UrbanSound8K MVP 快速執行

本專案目前的程式主線是將 UrbanSound8K 音訊轉成 Mel-spectrogram，再用 CNN baseline 與 Spectrogram Transformer 進行 10 類聲音分類。

安裝環境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

正式資料需放在：

```text
data/raw/UrbanSound8K/
├── audio/
│   ├── fold1/
│   └── ...
└── metadata/
    └── UrbanSound8K.csv
```

先產生 Mel-spectrogram：

```bash
python3 -m src.preprocess \
  --raw-dir data/raw/UrbanSound8K \
  --out-dir data/processed/urbansound8k_mels
```

訓練 CNN baseline：

```bash
python3 -m src.train --config configs/cnn_baseline.yaml --fold 10
```

訓練 Spectrogram Transformer：

```bash
python3 -m src.train --config configs/transformer_baseline.yaml --fold 10
```

重新評估已訓練 run：

```bash
python3 -m src.evaluate --run-dir results/transformer_baseline_fold10
```

如果暫時沒有 UrbanSound8K，可先建立 synthetic smoke-test dataset：

```bash
python3 scripts/create_synthetic_urbansound8k.py --out-dir data/raw/UrbanSound8K_synthetic
python3 -m src.preprocess --raw-dir data/raw/UrbanSound8K_synthetic --out-dir data/processed/urbansound8k_synthetic_mels
```

## 重要文件入口

- `docs/intake_questions.md`：還不知道怎麼開始時，先回答這份訪談表。
- `docs/next_input_template.md`：下一次給 AI Agent 的可複製填空範本。
- `docs/dashboard.md`：專案儀表板，快速查看目前階段、最高風險與下一步。
- `docs/current_status.md`：目前完成度、資訊缺口與下一步。
- `docs/first_week_plan.md`：第一週啟動計畫，把資訊缺口轉成任務。
- `docs/professor_questions.md`：與教授確認研究、程式、資料、格式和時程的問題清單。
- `docs/professor_update_template.md`：給教授的更新、詢問、會前確認與會後整理模板。
- `docs/thesis_plan.md`：整理論文題目、研究問題、方法與預期成果。
- `docs/thesis_outline.md`：建立論文章節架構與各章待補內容。
- `docs/chapters/`：逐章撰寫正式草稿。
- `docs/writing_style_guide.md`：論文語氣、段落、引用與 AI 改稿原則。
- `docs/glossary.md`：統一術語、縮寫與翻譯決策。
- `docs/task_inbox.md`：暫存尚未分類的論文、程式、資料、文獻任務。
- `docs/decision_log.md`：記錄重要研究與技術決策，以及決策理由。
- `docs/artifact_index.md`：追蹤所有重要成果、狀態、證據來源與對應章節。
- `docs/risk_register.md`：追蹤論文、程式、資料、時程與交付風險。
- `docs/milestones.md`：追蹤論文、程式、實驗、教授回饋與期限。
- `docs/weekly_review.md`：每週檢查論文、程式、資料、文獻與風險。
- `docs/meeting_notes.md`：記錄教授會議、修改要求與下次會議前任務。
- `docs/ai_workflow.md`：說明如何讓 AI Agent 協助寫作、文獻與程式。
- `docs/ai_usage_log.md`：記錄 AI Agent 協助了哪些重要工作。
- `docs/ai_disclosure_draft.md`：若需揭露 AI 使用，可作為草稿。
- `docs/environment.md`：記錄程式語言、工具、安裝命令與重現步驟。
- `docs/code_task_spec.md`：每個程式任務的目的、輸入、輸出、驗證方式與對應章節。
- `docs/experiment_template.md`：每次實驗或跑程式後的紀錄格式。
- `docs/reproducibility_checklist.md`：交付前檢查研究是否可追蹤、可重現。
- `docs/submission_checklist.md`：交付教授、階段成果或初稿前的檢查清單。
- `scripts/check_project_status.py`：快速檢查關鍵文件與待補標記。
- `.env.example`：環境變數樣板，實際 `.env` 不提交。
- `references/README.md`：文獻與引用管理流程。
- `references/literature_notes.md`：文獻閱讀與引用重點。
- `references/citation_tracker.md`：追蹤各章節需要補哪些引用。
- `references/references.bib`：BibTeX 參考文獻檔。
- `src/README.md`：正式程式碼的放置與整理原則。
- `notebooks/README.md`：探索性 notebook 的使用原則。
- `data/README.md`：資料來源、資料處理與隱私限制的紀錄方式。
- `results/README.md`：實驗輸出與結果紀錄方式。
- `figures/README.md`：論文圖表命名、追溯與圖說紀錄方式。

## 下一步

目前最重要的下一步：

1. 下載 UrbanSound8K 並放到 `data/raw/UrbanSound8K/`。
2. 跑 preprocessing。
3. 先用 `--fold 10` 跑 CNN baseline 與 Transformer。
4. 將 metrics 與 confusion matrix 帶去和教授討論。
