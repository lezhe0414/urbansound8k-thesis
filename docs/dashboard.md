# 專案儀表板

更新日期：2026-06-29

## 目前階段

研究方向已確認，已進入可展示 MVP 與正式實驗補強階段。

Project definition 已確認方向為「Sound Event Detection Using Machine Learning Techniques」。核心技術路線是將音訊轉成 Mel-spectrogram 等頻譜圖，再以 CNN 進行聲音事件分類。可用公開資料集包含 UrbanSound8K 或 ESC-50，工具以 Python、PyTorch、Librosa、NumPy、Matplotlib 為主。

目前進度仍落後於原時程，但端到端 pipeline 已能運作：UrbanSound8K 已下載驗證，已轉成 Mel-spectrogram，CNN baseline 與 Spectrogram Transformer 的 smoke run 已輸出 metrics 與 confusion matrix；Spectrogram Transformer fold 10 已完成正式訓練，test accuracy 約 `0.655`、macro F1 約 `0.664`。下一步應優先補 CNN baseline 正式結果，必要時改用 Colab/GPU，而不是再擴大模型範圍。

## 立即執行方向

1. 保留 CNN baseline，不把 CNN 從論文中拿掉。
2. 使用 Spectrogram Transformer 作為現代比較模型，和 CNN baseline 形成清楚對照。
3. 用 `configs/*_smoke.yaml` 展示 pipeline 已跑通；已完成 `configs/transformer_baseline.yaml` 正式結果，接著補 `configs/cnn_baseline.yaml`。
4. 將每次實驗結果保存到 `results/`，圖表保存到 `figures/`。
5. 需要向教授確認：是否接受將 Transformer 作為 CNN baseline 之外的比較模型。

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

1. 跑正式 CNN baseline 訓練，必要時改用 Colab/GPU。
2. 整理 Transformer metrics 與 confusion matrix，準備教授討論。
3. 若 CNN 來不及正式長訓練，明確標示 CNN 目前是 baseline architecture/smoke result，Transformer 是已完成正式 fold 10 結果。
4. 撰寫方法章草稿，說明 audio-to-spectrogram 與模型比較。
5. 準備週五教授討論重點。

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
- 實驗結果：smoke run 已完成，Transformer fold 10 正式結果已完成，CNN 正式長訓練待補。

目前不可將整體目標標記為完成，因為仍需完成正式實驗、圖表解讀與 8 頁論文。
