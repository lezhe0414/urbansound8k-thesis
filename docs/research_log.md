# 研究日誌

每次討論、實驗、寫作或重要決策都可以記在這裡，避免之後忘記脈絡。

## 2026-06-28

### 事件

- 建立論文專案骨架。
- 設定 AI Agent 協作規範。
- 新增論文需求訪談表 `docs/intake_questions.md`。
- 新增里程碑與任務追蹤文件 `docs/milestones.md`。
- 新增程式碼目錄說明 `src/README.md`。
- 新增 notebook 使用說明 `notebooks/README.md`。
- 新增實驗紀錄模板 `docs/experiment_template.md`。
- 新增可重複研究檢查清單 `docs/reproducibility_checklist.md`。
- 新增論文章節大綱 `docs/thesis_outline.md`。
- 新增教授會議紀錄 `docs/meeting_notes.md`。
- 新增資料、結果與圖表資料夾說明。
- 新增決策紀錄 `docs/decision_log.md`。
- 新增任務 Inbox `docs/task_inbox.md`。
- 新增程式環境設定 `docs/environment.md` 與 `.env.example`。
- 新增逐章論文草稿資料夾 `docs/chapters/`。
- 新增引用管理入口 `references/README.md`、`references/citation_tracker.md`、`references/references.bib`。
- 新增目前狀態摘要 `docs/current_status.md`。
- 新增下一次輸入範本 `docs/next_input_template.md`。
- 新增每週進度檢查 `docs/weekly_review.md`。
- 新增第一週啟動計畫 `docs/first_week_plan.md`。
- 新增教授問題清單 `docs/professor_questions.md`。
- 新增成果索引 `docs/artifact_index.md`。
- 新增交付檢查清單 `docs/submission_checklist.md`。
- 新增程式任務規格模板 `docs/code_task_spec.md`。
- 新增 AI Agent 使用紀錄 `docs/ai_usage_log.md`。
- 新增 AI 使用揭露草稿 `docs/ai_disclosure_draft.md`。
- 新增論文寫作風格指南 `docs/writing_style_guide.md`。
- 新增術語表 `docs/glossary.md`。
- 新增風險與阻塞追蹤 `docs/risk_register.md`。
- 新增專案儀表板 `docs/dashboard.md`。
- 新增給教授的更新與詢問模板 `docs/professor_update_template.md`。
- 新增專案狀態檢查腳本 `scripts/check_project_status.py`。
- 讀取 project definition PDF，確認題目為 sound event detection using machine learning techniques。
- 確認 project definition 的主路線為 audio-to-spectrogram preprocessing 與 CNN classification。
- 根據目前落後時程，將下一步收斂為資料處理與 CNN baseline。
- 記錄 Transformer 策略：不直接取代 CNN，先作為可選比較模型。
- 建立 UrbanSound8K Mel-spectrogram classification MVP 程式架構。
- 新增 CNN baseline 與 Spectrogram Transformer 模型。
- 新增 preprocessing、training、evaluation 命令。
- 新增 synthetic UrbanSound8K-like dataset 產生器，方便沒有正式資料時做 smoke test。

### 已知資訊

- 專案目標是協助完成論文寫作與程式碼開發。
- 指導教授已同意使用 AI Agent。
- 論文題目是 Sound Event Detection Using Machine Learning Techniques。
- 原定 timeplan 要求 late June 完成 baseline CNN initial training 與 preliminary results。
- 可用資料集包含 UrbanSound8K 或 ESC-50。
- 程式工具以 Python、PyTorch、Librosa、NumPy、Matplotlib 為主。

### 待補資訊

- 具體選用 UrbanSound8K 或 ESC-50。
- 教授是否接受 Transformer 作為額外比較模型。
- 教授對論文格式、進度和實驗的要求。

### 下一步

- 建立資料處理與 CNN baseline 程式碼。
- 先查看 `docs/dashboard.md`，快速確認目前階段、最高風險與下一步。
- 先查看 `docs/current_status.md`，確認仍缺哪些資訊。
- 依 `docs/first_week_plan.md` 完成第一週啟動任務。
- 使用 `docs/professor_questions.md` 準備下一次要向教授確認的問題。
- 使用 `docs/professor_update_template.md` 整理可傳給教授的訊息。
- 使用 `docs/next_input_template.md` 收集論文方向、程式需求與教授要求。
- 用 `docs/artifact_index.md` 追蹤新增成果。
- 用 `docs/risk_register.md` 追蹤高風險資訊缺口與阻塞。
- 依 `docs/thesis_outline.md` 建立第一版正式章節架構。
- 開始在 `docs/chapters/` 補第一章緒論或第二章文獻探討。
- 開始撰寫前，依 `docs/writing_style_guide.md` 與 `docs/glossary.md` 統一語氣與術語。
- 把教授既有要求補到 `docs/meeting_notes.md`。
- 把未分類任務先放到 `docs/task_inbox.md`。
- 把重要研究或技術選擇補到 `docs/decision_log.md`。
- 先回答 `docs/intake_questions.md` 最後三題，以便 AI Agent 整理研究計畫。
- 使用 `docs/code_task_spec.md` 的規格開始撰寫 preprocessing、training、evaluation。
- 決定是否先使用 UrbanSound8K 或 ESC-50。
- 下載 UrbanSound8K 正式資料並放入 `data/raw/UrbanSound8K/`。
- 跑 `python3 -m src.preprocess` 產生 Mel-spectrogram cache。
- 分別跑 CNN baseline 與 Spectrogram Transformer 的 fold 10 訓練。
- 確認學校或教授要求的引用格式。
- 每週使用 `docs/weekly_review.md` 檢查進度與風險。
- 可執行 `python3 scripts/check_project_status.py` 快速檢查關鍵文件與待補標記；必要時使用 `--write-report` 保存報告。
- 交付教授或階段成果前使用 `docs/submission_checklist.md`。
- 重要 AI 協助需記錄到 `docs/ai_usage_log.md`，是否正式揭露需向教授確認。
