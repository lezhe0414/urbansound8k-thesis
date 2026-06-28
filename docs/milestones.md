# 論文里程碑與任務追蹤

這份文件用來追蹤論文寫作、程式開發、實驗與教授回饋。

## 目前狀態

- 專案建立日期：2026-06-28
- 目前階段：研究方向已確認，進入程式原型
- 下一個主要目標：完成 sound event detection 的資料處理與 CNN baseline

## 里程碑

| 狀態 | 里程碑 | 預計日期 | 產出 | 備註 |
| --- | --- | --- | --- | --- |
| 已確認 | 確認論文題目與研究問題 | 2026-06-28 | `docs/thesis_plan.md` | 來自 project definition |
| 未開始 | 完成初版文獻清單 | 2026-07-02 | `references/literature_notes.md` | sound event detection、spectrogram CNN、可選 Transformer |
| 未開始 | 確認資料來源與資料格式 | 2026-06-29 | `data/` 說明或資料樣本 | 優先 UrbanSound8K 或 ESC-50 |
| 草稿 | 建立資料處理 pipeline | 2026-06-30 | `src/`、`data/processed/` | Mel-spectrogram preprocessing 已有程式，待正式資料驗證 |
| 草稿 | 建立 CNN baseline | 2026-07-02 | `src/`、`results/` | 模型與訓練程式已建立，待正式資料訓練 |
| 未開始 | 完成第一輪實驗 | 2026-07-05 | `results/`、`figures/` | metrics + confusion matrix |
| 未開始 | 完成 draft dissertation 內容骨架 | 2026-07-09 | `docs/chapters/` | 原定 draft submission |
| 草稿 | 加入 Transformer 或 transfer learning 比較 | 2026-07-15 | `src/`、`results/` | Spectrogram Transformer 已建立，待正式資料訓練 |
| 未開始 | 完成結果與討論草稿 | 2026-08-05 | `docs/` 草稿 | 根據實驗結果撰寫 |
| 未開始 | 完成 8 頁 final PDF | 2026-08-19 | 完整 PDF | final submission |

## 任務清單

### 立即任務

- [x] 先查看 `docs/current_status.md`，確認目前資訊缺口。
- [x] 選定 UrbanSound8K。
- [ ] 建立資料下載或放置說明。
- [x] 建立 Mel-spectrogram preprocessing。
- [x] 建立 CNN baseline training。
- [x] 建立 evaluation + confusion matrix 輸出。
- [ ] 使用 `docs/professor_questions.md` 準備 CNN/Transformer 的確認問題。
- [ ] 把零散想法先整理到 `docs/task_inbox.md`。
- [ ] 用 `docs/artifact_index.md` 追蹤新增成果。
- [ ] 用 `docs/risk_register.md` 追蹤目前高風險資訊缺口。
- [x] 補齊 `docs/thesis_plan.md` 的基本資訊。
- [ ] 依 `docs/thesis_outline.md` 建立第一版章節大綱。
- [ ] 列出教授已經給的要求。
- [ ] 把教授近期要求整理到 `docs/meeting_notes.md`。
- [ ] 將已確認的重要決策記到 `docs/decision_log.md`。
- [x] 決定第一個要實作的程式任務。
- [ ] 建立第一次 `docs/weekly_review.md` 週檢查紀錄。
- [ ] 交付任何內容前使用 `docs/submission_checklist.md`。

### 寫作任務

- [ ] 建立論文章節大綱。
- [ ] 確認學校或教授要求的正式章節格式。
- [ ] 在 `docs/chapters/` 建立並維護逐章草稿。
- [ ] 整理研究背景。
- [ ] 整理研究問題與研究貢獻。
- [ ] 建立文獻探討架構。
- [ ] 撰寫研究方法。
- [ ] 撰寫結果與討論。
- [ ] 撰寫結論與未來工作。

### 程式任務

- [x] 決定使用的程式語言與環境。
- [x] 用 `docs/code_task_spec.md` 定義第一個程式任務。
- [x] 補上 `docs/environment.md` 的安裝與執行方式。
- [ ] 依 `src/README.md` 決定正式程式碼結構。
- [ ] 依 `notebooks/README.md` 建立探索性分析規則。
- [ ] 依 `data/README.md` 記錄資料來源與使用限制。
- [x] 建立可重複執行的資料處理流程。
- [x] 建立實驗或模型腳本。
- [ ] 每次實驗依 `docs/experiment_template.md` 留下紀錄。
- [ ] 用 `docs/reproducibility_checklist.md` 檢查程式與實驗是否可重現。
- [ ] 儲存結果與圖表。
- [ ] 依 `figures/README.md` 記錄論文圖表來源與圖說。
- [ ] 撰寫執行說明。

### 教授回饋

| 日期 | 教授意見 | 需要修改的地方 | 狀態 |
| --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待填 |

### 文獻與引用任務

- [ ] 確認引用格式。
- [ ] 將第一批核心文獻整理到 `references/literature_notes.md`。
- [ ] 將正文待補引用整理到 `references/citation_tracker.md`。
- [ ] 建立或匯入 `references/references.bib`。
- [ ] 定稿前確認正文引用與參考文獻清單一致。

## 每週檢查

每週至少確認一次：

- 目前論文最重要的阻塞點是什麼？
- 本週是否有新增文獻、程式、實驗結果或章節草稿？
- 下週是否有教授會議或截止日期？
- 哪個任務最能推進畢業進度？
