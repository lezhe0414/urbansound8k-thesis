# 專案儀表板

更新日期：2026-08-17

## 目前階段

研究方向已確認，已進入可展示 MVP 與正式實驗補強階段。

Project definition 已確認方向為「Sound Event Detection Using Machine Learning Techniques」。核心技術路線是將音訊轉成 Mel-spectrogram 等頻譜圖，再以 CNN 進行聲音事件分類。可用公開資料集包含 UrbanSound8K 或 ESC-50，工具以 Python、PyTorch、Librosa、NumPy、Matplotlib 為主。

端到端 pipeline、CNN validation-only augmentation 搜尋及鎖定設定的正式 10-fold 均已完成。From-scratch CNN 的正式 Macro F1 為 `0.79041 ± 0.04755`、Accuracy 為 `0.77423 ± 0.05431`。AudioSet-pretrained EfficientAT MN20 在相同 ten-fold coverage 下達 `0.87686 ± 0.04048`，是目前最強正式模型。單一 fold 10 CNN 分數 `0.8536` 高於十折平均，不能再作 headline result。

## 立即執行方向

1. 以 from-scratch CNN 與 pretrained MN20 的 formal 10-fold mean/std 作主要比較。
2. 不得再依 fold 10、ensemble test 或 10-fold 結果調整超參數。
3. 從零訓練 Transformer 保留為 fold-10-only 架構比較；不可假裝已有 matched 10-fold。
4. EMA 與 3-seed ensemble 作為負結果；`0.90104` cross-scale ensemble 作為 post-formal development-only 結果。
5. 完成英文 Results/Discussion、混淆矩陣、Harvard references 與 8 頁 PDF。

## 最高風險

目前最高風險記錄在 `docs/risk_register.md`：

| ID | 風險 | 等級 | 下一步 |
| --- | --- | --- | --- |
| R-001 | 進度已落後原 timeplan | 高 | 用已跑通 MVP 支撐週五討論，正式訓練改用 Colab/GPU |
| R-002 | Transformer 可能偏離原 definition 的 CNN 承諾 | 中 | 保留 CNN baseline，Transformer 作比較模型 |
| R-003 | 本機 CPU 訓練 CNN 偏慢 | 中 | smoke run 已完成，正式訓練建議用 Colab/GPU |
| R-004 | 8 頁論文空間有限 | 中 | 聚焦一個資料集、少量模型、清楚評估 |

若需要向教授確認上述事項，可使用 `docs/professor_update_template.md`。

## 下一步工作流

下一步 AI Agent 應依序執行：

1. 將已驗證的 formal 10-fold table 與 per-class error analysis 移入論文。
2. 將單一 CNN、pretrained MN20、Transformer 與 exploratory ensembles 分層比較。
3. 加入 aggregate confusion matrix 與對 machinery-class confusion 的討論。
4. 更新英文 PDF，避免把單一 fold 或 development-only 結果描述成正式結論。

文獻與寫作同步更新：

1. `references/literature_notes.md`
2. `references/citation_tracker.md`
3. `references/references.bib`

## 關鍵入口

| 目的 | 文件 |
| --- | --- |
| 看目前完成度 | `docs/current_status.md` |
| 看 MVP 已做 / 未做 checklist | `docs/progress_tracker.md` |
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

若要保存 Markdown 報告：

```text
python3 scripts/check_project_status.py --write-report
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
- 正式論文內容：方向已確認，正文尚未開始。
- 實際程式碼：MVP 已完成。
- 實驗結果：from-scratch CNN 與 pretrained MN20 正式 10-fold 已完成；Transformer 目前只有 fold 10。

目前不可將整體目標標記為完成，因為仍需完成圖表整合、引用核對與 8 頁論文定稿。
