# 專案儀表板

更新日期：2026-06-28

## 目前階段

研究方向已確認，進入程式原型與補進度階段。

Project definition 已確認方向為「Sound Event Detection Using Machine Learning Techniques」。核心技術路線是將音訊轉成 Mel-spectrogram 等頻譜圖，再以 CNN 進行聲音事件分類。可用公開資料集包含 UrbanSound8K 或 ESC-50，工具以 Python、PyTorch、Librosa、NumPy、Matplotlib 為主。

目前進度落後於原時程：definition 預期 6 月底已有 baseline CNN 初步結果，現在應優先補上可重複執行的資料處理與 baseline 模型，而不是先擴大論文範圍。

## 立即執行方向

1. 建立音訊資料處理流程：載入資料、切分 train/validation/test、產生 Mel-spectrogram。
2. 建立 baseline CNN：先完成能訓練、評估、輸出指標與 confusion matrix 的最小版本。
3. 若 baseline 可運作，再加入 Transformer 或 transfer learning 模型作比較，不建議直接放棄 CNN。
4. 將每次實驗結果保存到 `results/`，圖表保存到 `figures/`。
5. 需要向教授確認：是否接受將 Transformer 作為 CNN baseline 之外的比較模型。

## 最高風險

目前最高風險記錄在 `docs/risk_register.md`：

| ID | 風險 | 等級 | 下一步 |
| --- | --- | --- | --- |
| R-001 | 進度已落後原 timeplan | 高 | 先做最小可交付：資料處理 + CNN baseline |
| R-002 | Transformer 可能偏離原 definition 的 CNN 承諾 | 中 | 保留 CNN baseline，Transformer 作比較模型 |
| R-003 | 資料集尚未下載與驗證 | 高 | 先選 UrbanSound8K 或 ESC-50 其中一個 |
| R-004 | 8 頁論文空間有限 | 中 | 聚焦一個資料集、少量模型、清楚評估 |

若需要向教授確認上述事項，可使用 `docs/professor_update_template.md`。

## 下一步工作流

下一步 AI Agent 應依序執行：

1. 建立 Python 專案依賴與執行方式。
2. 建立資料處理腳本。
3. 建立 CNN baseline 訓練腳本。
4. 建立評估與圖表輸出腳本。
5. 若時間允許，建立 Transformer/ViT-like 或 audio spectrogram transformer 比較實驗。

文獻與寫作同步更新：

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
- 實際程式碼：尚未開始。
- 實驗結果：尚未開始。

目前不可將整體目標標記為完成，因為仍需完成程式、實驗、圖表與 8 頁論文。
