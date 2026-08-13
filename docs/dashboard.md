# 專案儀表板

更新日期：2026-08-13

## 目前階段

研究方向已確認，已進入可展示 MVP 與正式實驗補強階段。

Project definition 已確認方向為「Sound Event Detection Using Machine Learning Techniques」。核心技術路線是將音訊轉成 Mel-spectrogram 等頻譜圖，再以 CNN 進行聲音事件分類。可用公開資料集包含 UrbanSound8K 或 ESC-50，工具以 Python、PyTorch、Librosa、NumPy、Matplotlib 為主。

端到端 pipeline 與 CNN 受控資料增強搜尋已完成。搜尋只依 validation Macro F1 選模，最佳值為 `0.7924`；鎖定唯一設定後的一次 fold 10 test Accuracy 為 `0.8471`、Macro F1 為 `0.8536`。相較歷史 Macro F1 `0.8413` 約提高 `0.0123`，但仍需固定設定的 10-fold mean/std 才能判斷泛化是否穩定。

## 立即執行方向

1. 鎖定 `configs/cnn_aug_final.yaml`，不得再依 fold 10 test 調參。
2. 在 Colab 執行正式 10-fold cross-validation，輸出 mean/std 與 per-class 指標。
3. CNN 作為主要模型；從零訓練 Transformer 作為架構比較；pretrained AST 作為 transfer-learning 延伸。
4. 實驗 artifacts 備份至 Google Drive，GitHub 只提交程式碼、設定與文件。
5. 將受控搜尋方法、停止條件與單一 fold 限制如實寫入論文。

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

1. 跑固定 `configs/cnn_aug_final.yaml` 的 10-fold cross-validation。
2. 整理 mean/std、每類 F1 與 aggregate confusion matrix。
3. 更新論文 Results、Discussion 與 limitations，避免把單一 fold 改善描述成確定結論。
4. 完成 pretrained AST 的快速可行性測試，再決定是否納入最終比較表。

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
