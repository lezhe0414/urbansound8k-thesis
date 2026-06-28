# 專案儀表板

更新日期：2026-06-28

## 目前階段

起始規劃。

專案基礎設施已建立，包含論文規劃、章節草稿、程式碼規範、資料與實驗管理、文獻引用、AI 使用紀錄、風險管理與交付檢查。正式論文內容與程式實作尚未開始，因為仍缺研究方向、程式需求、資料來源與教授近期要求。

## 立即需要使用者補充

請先補齊以下三句：

```text
我的論文大方向是：
程式需要做的是：
教授最近要求我完成的是：
```

可直接複製 `docs/next_input_template.md` 的建議版本填寫。

## 最高風險

目前最高風險記錄在 `docs/risk_register.md`：

| ID | 風險 | 等級 | 下一步 |
| --- | --- | --- | --- |
| R-001 | 尚未確認論文題目或研究方向 | 高 | 使用者補上論文大方向 |
| R-002 | 尚未確認程式需要完成的功能 | 高 | 使用者或教授確認程式需求 |
| R-003 | 尚未確認教授近期要求 | 高 | 補上教授最近要求 |
| R-004 | 尚未確認資料來源與使用限制 | 高 | 確認資料來源或樣本 |

若需要向教授確認上述事項，可使用 `docs/professor_update_template.md`。

## 下一步工作流

取得使用者或教授資訊後，AI Agent 應依序更新：

1. `docs/intake_questions.md`
2. `docs/thesis_plan.md`
3. `docs/current_status.md`
4. `docs/risk_register.md`
5. `docs/milestones.md`

若已有程式方向，接著更新：

1. `docs/code_task_spec.md`
2. `docs/environment.md`
3. `src/` 或 `notebooks/`

若已有文獻或教授指定閱讀，接著更新：

1. `references/literature_notes.md`
2. `references/citation_tracker.md`
3. `references/references.bib`

## 關鍵入口

| 目的 | 文件 |
| --- | --- |
| 看目前完成度 | `docs/current_status.md` |
| 直接提供下一步資訊 | `docs/next_input_template.md` |
| 第一週啟動 | `docs/first_week_plan.md` |
| 準備問教授 | `docs/professor_questions.md` |
| 傳給教授的訊息模板 | `docs/professor_update_template.md` |
| 追蹤風險 | `docs/risk_register.md` |
| 追蹤所有成果 | `docs/artifact_index.md` |
| 追蹤任務與里程碑 | `docs/milestones.md` |
| 寫論文章節 | `docs/chapters/` |
| 定義程式任務 | `docs/code_task_spec.md` |
| 管理文獻引用 | `references/` |
| 交付前檢查 | `docs/submission_checklist.md` |
| 自動檢查狀態 | `scripts/check_project_status.py` |

## 快速檢查命令

```text
python3 scripts/check_project_status.py
```

## 最近提交

```text
b1f8459 Add thesis risk register
63ff1c9 Add thesis writing style and glossary templates
39130e8 Add AI usage tracking templates
9c0a0a0 Add code task specification template
3d58256 Add artifact index and submission checklist
```

## 完成度判斷

- 專案協作骨架：已建立。
- 論文規劃模板：已建立。
- 程式與實驗模板：已建立。
- 風險與交付管理：已建立。
- 正式論文內容：尚未開始。
- 實際程式碼：尚未開始。
- 實驗結果：尚未開始。

目前不可將整體目標標記為完成，因為仍需使用者提供論文方向、程式需求、教授要求與資料來源。
