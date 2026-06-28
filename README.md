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
7. 不確定要放哪裡的事項，先放進 `docs/task_inbox.md`。
8. 重要方向、技術選型或教授要求改變時，記到 `docs/decision_log.md`。
9. 用 `docs/artifact_index.md` 追蹤所有重要成果。
10. 用 `docs/risk_register.md` 追蹤風險與阻塞。
11. 用 `docs/milestones.md` 追蹤論文、程式、實驗與教授回饋。
12. 每次與教授討論後，把結論寫進 `docs/research_log.md` 與 `docs/meeting_notes.md`。
13. 重要 AI 協助記錄到 `docs/ai_usage_log.md`。
14. 寫作時依 `docs/writing_style_guide.md` 與 `docs/glossary.md` 統一語氣和術語。
15. 每週用 `docs/weekly_review.md` 檢查進度與風險。
16. 章節正式草稿放在 `docs/chapters/`。
17. 寫程式前先用 `docs/code_task_spec.md` 定義任務規格。
18. 程式原型可先放 `notebooks/`，確定要重複執行後整理到 `src/`。
19. 程式環境與執行方式記在 `docs/environment.md`。
20. 每次實驗前後參考 `docs/experiment_template.md` 紀錄目的、命令、參數與結果。
21. 用 `docs/reproducibility_checklist.md` 檢查程式、資料與實驗是否能支撐論文。
22. 交給教授或階段交付前，用 `docs/submission_checklist.md` 檢查。
23. 每次實驗輸出放到 `results/`，圖表放到 `figures/`。
24. 每篇重要文獻都在 `references/literature_notes.md` 留下摘要、方法、可引用觀點與疑問，引用需求追蹤在 `references/citation_tracker.md`。

## 重要文件入口

- `docs/intake_questions.md`：還不知道怎麼開始時，先回答這份訪談表。
- `docs/next_input_template.md`：下一次給 AI Agent 的可複製填空範本。
- `docs/dashboard.md`：專案儀表板，快速查看目前階段、最高風險與下一步。
- `docs/current_status.md`：目前完成度、資訊缺口與下一步。
- `docs/first_week_plan.md`：第一週啟動計畫，把資訊缺口轉成任務。
- `docs/professor_questions.md`：與教授確認研究、程式、資料、格式和時程的問題清單。
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

請先補充或交給 AI Agent 協助整理以下資訊：

- 論文題目或暫定方向
- 系所與學位要求
- 教授目前給的研究範圍
- 需要寫的程式類型
- 資料來源
- 預計使用的語言與工具
- 截止日期與近期里程碑

如果不知道如何開始，先回答 `docs/intake_questions.md` 最後的三題：

1. 我的論文題目或大方向是什麼？
2. 我需要寫的程式大概是什麼？
3. 教授最近要求我完成的是什麼？
