# 專案目前狀態

更新日期：2026-06-28

## 專案目的

本專案用來協助完成論文寫作、程式碼開發、資料整理、實驗紀錄、圖表產出與教授回饋追蹤。

教授已同意使用 AI Agent，因此本專案已把 AI Agent 協作流程納入正式工作方式。

## 已建立內容

### 協作規範

- `AGENTS.md`：AI Agent 協作規則。
- `README.md`：專案入口與工作流程。
- `docs/ai_workflow.md`：如何讓 AI Agent 協助寫作、文獻、程式與實驗。

### 論文規劃

- `docs/intake_questions.md`：論文需求訪談表。
- `docs/next_input_template.md`：下一次給 AI Agent 的填空範本。
- `docs/first_week_plan.md`：第一週啟動計畫。
- `docs/professor_questions.md`：要向教授確認的問題。
- `docs/thesis_plan.md`：論文計畫。
- `docs/thesis_outline.md`：論文章節大綱。
- `docs/chapters/`：逐章草稿。

### 進度與決策管理

- `docs/milestones.md`：里程碑與任務追蹤。
- `docs/weekly_review.md`：每週進度檢查。
- `docs/task_inbox.md`：未分類任務入口。
- `docs/decision_log.md`：重要決策紀錄。
- `docs/artifact_index.md`：成果索引。
- `docs/research_log.md`：研究日誌。
- `docs/meeting_notes.md`：教授會議紀錄。

### 程式與實驗

- `src/README.md`：正式程式碼放置規則。
- `notebooks/README.md`：探索性 notebook 規則。
- `docs/environment.md`：程式環境與執行方式。
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

## 尚未取得的必要資訊

目前尚未能開始撰寫正式內容或寫實際程式，因為缺少以下資訊：

1. 論文題目或暫定研究方向。
2. 學校、系所、學位與格式要求。
3. 教授目前給的研究範圍或近期要求。
4. 程式需要完成的功能。
5. 資料來源、資料格式與使用限制。
6. 預計使用的程式語言、框架或工具。
7. 近期截止日期與里程碑。

## 下一步

最有效的下一步是補齊以下三句：

```text
我的論文大方向是：
程式需要做的是：
教授最近要求我完成的是：
```

也可以直接複製 `docs/next_input_template.md` 裡的建議版本填寫。

如果需要先整理第一週該做什麼，請依 `docs/first_week_plan.md` 執行；如果需要和教授確認方向，請使用 `docs/professor_questions.md`。

取得上述資訊後，AI Agent 應優先執行：

1. 更新 `docs/intake_questions.md`。
2. 更新 `docs/thesis_plan.md`。
3. 更新 `docs/thesis_outline.md` 與 `docs/chapters/01_introduction.md`。
4. 若已有程式方向，更新 `docs/environment.md` 並建立第一個 `src/` 或 `notebooks/` 原型。
5. 若已有文獻或教授指定閱讀，更新 `references/literature_notes.md` 與 `references/citation_tracker.md`。
6. 若是每週整理，更新 `docs/weekly_review.md`、`docs/milestones.md` 與 `docs/research_log.md`。
7. 若取得教授回覆，更新 `docs/meeting_notes.md`、`docs/decision_log.md` 與 `docs/current_status.md`。
8. 若新增任何重要成果，更新 `docs/artifact_index.md`；交付前使用 `docs/submission_checklist.md`。

## 目前完成度判斷

專案基礎設施已完成，但論文內容與程式實作尚未開始。整體目標尚未完成，因為正式論文內容、研究方法、程式碼與實驗結果都需要使用者提供研究方向後才能落地。
