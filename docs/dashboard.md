# 專案儀表板

更新日期：2026-08-14

## 目前階段

研究方向已確認，已進入可展示 MVP 與正式實驗補強階段。

Project definition 已確認方向為「Sound Event Detection Using Machine Learning Techniques」。核心技術路線是將音訊轉成 Mel-spectrogram 等頻譜圖，再以 CNN 進行聲音事件分類。可用公開資料集包含 UrbanSound8K 或 ESC-50，工具以 Python、PyTorch、Librosa、NumPy、Matplotlib 為主。

端到端 pipeline 與 CNN 受控資料增強搜尋已完成。搜尋只依 validation Macro F1 選模，最佳值為 `0.7924`；鎖定唯一設定後的一次 fold 10 test Accuracy 為 `0.8471`、Macro F1 為 `0.8536`。EMA 與固定 3-seed probability ensemble 亦已完成，但均未超越單一 CNN。

`codex/cnn-breakthrough-90` 的高風險 validation-only study 亦已完成：五個候選在 folds 1、4、7 共執行 15 次。Cooldown 名義上以平均 Macro F1 `0.7821` 排名第一，但只比 control `0.7818` 高 `0.00025`，標準差反而由 `0.0044` 增至 `0.0104`，不足以視為穩健改善。此分支不合併回 `main`，fold 10 test 全程未評估。

## 立即執行方向

1. 回到 `main`，使用已鎖定且 EMA 關閉的 `configs/cnn_aug_final.yaml`。
2. 不再調整 CNN 設定，直接執行正式 10-fold cross-validation。
3. 整理 mean/std、per-class F1 與 aggregate confusion matrix，作為論文主要 CNN 結果。
4. CNN 作為主要模型；從零訓練 Transformer 作為架構比較；pretrained AST 作為 transfer-learning 延伸。
5. 實驗 artifacts 備份至 Google Drive，GitHub 只提交程式碼、設定與文件。

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

1. 保留 breakthrough 分支作負結果與重現紀錄，不合併候選程式到主線。
2. 以 `main` 的鎖定主模型執行正式 10-fold cross-validation，整理 mean/std、每類 F1 與 aggregate confusion matrix。
3. 比較單一 fold 與 10-fold 統計，檢查先前改善是否能跨 folds 泛化。
4. 將單一 CNN、突破候選、EMA、3-seed ensemble、Transformer 與 AST 的結果整理成公平比較。
5. 更新論文 Results、Discussion 與 limitations，避免把單一 fold 改善描述成確定結論。

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
- 實驗結果：CNN fold 10、受控 augmentation、EMA、3-seed ensemble、突破性三-fold development study 與 Transformer fold 10 均已完成；CNN 正式 10-fold 待補。

目前不可將整體目標標記為完成，因為仍需完成正式實驗、圖表解讀與 8 頁論文。
