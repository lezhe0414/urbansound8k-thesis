# 成果索引

這份文件用來追蹤本專案所有重要產出，包含論文章節、程式碼、資料、實驗結果、圖表、文獻與決策。當專案內容變多時，先看這份文件可以知道每個成果目前在哪裡、狀態如何、支撐哪個論文部分。

## 使用方式

- 新增重要文件、程式、資料、結果或圖表時，在對應表格新增一列。
- 每次成果狀態改變時，更新「狀態」。
- 若成果支撐論文主張，務必填寫「對應章節」與「證據來源」。
- 若成果還沒被驗證，狀態不要標成完成。

## 狀態定義

- `待補`：尚未建立內容。
- `草稿`：已有初版，但未經檢查或教授確認。
- `進行中`：正在撰寫、開發、整理或實驗。
- `待確認`：需要使用者、教授或資料來源確認。
- `已確認`：已經確認可用，但可能還未定稿。
- `完成`：已完成且有足夠證據支撐。

## 論文文件

| 成果 | 路徑 | 對應章節 | 狀態 | 證據來源 | 備註 |
| --- | --- | --- | --- | --- | --- |
| 論文計畫 | `docs/thesis_plan.md` | 全文 | 草稿 | Project definition PDF | 已補 sound event detection 方向 |
| 論文章節大綱 | `docs/thesis_outline.md` | 全文 | 草稿 | 通用章節架構 | 需依學校格式調整 |
| 寫作風格指南 | `docs/writing_style_guide.md` | 全文 | 已確認 | 文件檢查 | 需依學校格式調整 |
| 術語表 | `docs/glossary.md` | 全文 | 草稿 | 研究領域術語 | 需依題目補充 |
| 摘要 | `docs/chapters/00_abstract.md` | 摘要 | 待補 | 研究結果 | 建議最後撰寫 |
| 緒論 | `docs/chapters/01_introduction.md` | 第一章 | 待補 | 研究方向、文獻 | 已取得題目，待寫正文 |
| 文獻探討 | `docs/chapters/02_literature_review.md` | 第二章 | 待補 | 文獻筆記 | 需核心文獻 |
| 研究方法 | `docs/chapters/03_methodology.md` | 第三章 | 待補 | 程式與資料流程 | 需方法與資料 |
| 系統或實驗設計 | `docs/chapters/04_system_or_experiment_design.md` | 第四章 | 待補 | 程式與實驗設定 | 需確認論文類型 |
| 結果與討論 | `docs/chapters/05_results_and_discussion.md` | 第五章 | 待補 | `results/`、`figures/` | 需實驗結果 |
| 結論 | `docs/chapters/06_conclusion.md` | 第六章 | 待補 | 最終結果與討論 | 建議最後撰寫 |

## 程式碼

| 成果 | 路徑 | 用途 | 狀態 | 驗證方式 | 備註 |
| --- | --- | --- | --- | --- | --- |
| 程式碼目錄規則 | `src/README.md` | 正式程式碼管理 | 已確認 | 文件檢查 | 已有 MVP 實作 |
| 專案狀態檢查腳本 | `scripts/check_project_status.py` | 檢查關鍵文件與待補標記 | 已確認 | 執行腳本 | 可用 `--write-report` 輸出報告 |
| Notebook 規則 | `notebooks/README.md` | 探索性分析管理 | 已確認 | 文件檢查 | 已有 Colab 執行 notebook |
| Colab CNN baseline notebook | `notebooks/2026-07-02-colab-cnn-baseline.ipynb` | 在 Colab GPU 執行 CNN fold 10 正式訓練 | 已確認 | notebook 文件檢查 | 含英文註解；GitHub 為程式碼來源，資料與輸出不提交 |
| Colab CNN + Transformer notebook | `notebooks/2026-07-08-colab-cnn-transformer-fold10.ipynb` | 在 Colab GPU 執行 CNN 與 Transformer fold 10 正式訓練 | 已確認 | notebook JSON 檢查 | 使用 Google Drive cache，輸出 fold 10 模型比較 artifacts |
| 程式環境設定 | `docs/environment.md` | 記錄安裝與執行方式 | 草稿 | Project definition PDF | Python/PyTorch/Librosa |
| 程式任務規格 | `docs/code_task_spec.md` | 定義 sound event detection baseline | 已確認 | Project definition PDF | MVP 已建立，待正式資料訓練 |
| CNN baseline | `src/models/cnn.py` | Sound event classification baseline | 已確認 | smoke run | 可訓練架構已建立，本機 CPU 較慢 |
| Transformer comparison | `src/models/spectrogram_transformer.py` | 現代模型比較 | 已確認 | smoke run | 作為主要比較模型 |
| Preprocessing pipeline | `src/preprocess.py` | Audio to Mel-spectrogram | 已確認 | UrbanSound8K preprocessing | 已處理 8732 筆 |
| Training pipeline | `src/train.py` | 模型訓練與測試 | 已確認 | smoke run | 產生 metrics 與 checkpoint |
| Evaluation pipeline | `src/evaluate.py` | 重跑評估與圖表 | 已確認 | smoke run evaluation | 產生 evaluation metrics |
| Smoke run configs | `configs/cnn_smoke.yaml`、`configs/transformer_smoke.yaml` | 本機快速驗證 | 已確認 | smoke run | 使用真實資料子集 |
| Spectrogram augmentation module | `src/data/augmentation.py` | CNN 訓練階段即時資料增強 | 已確認 | unit tests + smoke run | 支援 shift、stretch、gain、noise、SpecAugment、Mixup、CutMix |
| CNN augmentation ablation configs | `configs/cnn_aug_control.yaml`、`configs/cnn_aug_light.yaml`、`configs/cnn_aug_balanced.yaml`、`configs/cnn_aug_strong.yaml` | 比較增強強度 | 已確認 | validation-only selection | Strong profile 勝出；不需重新 preprocessing |
| CNN augmentation runner | `scripts/run_cnn_augmentation_ablation.py` | 依序訓練並用 validation Macro F1 排名 | 已確認 | CLI + smoke pipeline | 預設不執行 test |
| CNN controlled search runner | `scripts/run_cnn_controlled_search.py` | 初始四組比較、單一變因迭代、失敗重試、進度紀錄與 Drive 備份 | 已確認 | unit tests + plan-only | 唯一設定鎖定後才允許一次 test |
| CNN final augmentation config | `configs/cnn_aug_final.yaml` | 以鎖定設定執行正式 10-fold cross-validation | 已確認 | 與勝出 resolved config 核對 | 不得再依 test 或 fold 結果調參 |
| EMA model-weight utility | `src/utils/ema.py` | 維護訓練權重的指數移動平均副本 | 已確認 | `tests/test_ema.py` | 第一次更新直接同步；分別保存最佳 online 與 EMA checkpoint |
| CNN EMA candidate config | `configs/cnn_aug_ema.yaml` | 在相同訓練中比較 online 與 EMA validation Macro F1 | 已確認 | validation-only run | EMA 僅比 online 高約 0.00089，正式設定不採用 |
| CNN 3-seed ensemble runner | `src/ensemble.py` | 訓練三個固定 seeds 並平均 softmax probabilities | 已確認 | 22 unit tests + Colab run | EMA 與個別 seed test 強制關閉；支援鎖定後一次 ensemble test |
| CNN dynamic feature channels | `src/data/urbansound8k.py` | 動態組合 Mel、delta、delta-delta 三通道 | 待 GPU 驗證 | unit tests + breakthrough candidate | 不修改既有 `.npz` cache，不需重新 preprocessing |
| CNN augmentation cooldown | `src/utils/schedules.py`、`src/train.py` | 後期逐步降低 augmentation/Mixup 機率 | 待 GPU 驗證 | unit tests + history regularization scale | 只作用於訓練 batch |
| SE attention CNN | `src/models/cnn.py` | 高容量 channel-attention CNN 候選 | 待 GPU 驗證 | model shape test | 獨立突破分支，不取代目前最佳設定 |
| CNN breakthrough runner | `scripts/run_cnn_breakthrough_search.py` | 比較五候選於 folds 1、4、7 的平均 validation Macro F1 | 待 GPU 執行 | validation-only protocol tests | fold 10 鎖定；逐 run 寫入 progress 與 Drive backup |

## 資料

| 成果 | 路徑 | 用途 | 狀態 | 證據來源 | 備註 |
| --- | --- | --- | --- | --- | --- |
| 資料管理規則 | `data/README.md` | 資料來源與限制管理 | 已確認 | 文件檢查 | 尚未有資料 |
| 原始資料目錄 | `data/raw/UrbanSound8K_soundata/` | 保存 UrbanSound8K | 已確認 | `soundata.validate()` | 8732 個音訊檔，預設不提交資料檔 |
| 處理後資料目錄 | `data/processed/urbansound8k_mels/` | 保存 spectrogram features | 已確認 | `src.preprocess` | 8732 筆 Mel-spectrogram，預設不提交資料檔 |

## 實驗與結果

| 成果 | 路徑 | 用途 | 狀態 | 證據來源 | 備註 |
| --- | --- | --- | --- | --- | --- |
| 實驗紀錄模板 | `docs/experiment_template.md` | 記錄每次實驗 | 已確認 | 文件檢查 | 尚未有實驗 |
| CNN smoke result | `results/cnn_baseline_smoke_fold10/` | CNN pipeline 驗證 | 已確認 | `src.train` | 子集 smoke run，非正式分數 |
| Transformer smoke result | `results/transformer_baseline_smoke_fold10/` | Transformer pipeline 驗證 | 已確認 | `src.train` | 子集 smoke run，非正式分數 |
| Transformer fold 10 result | `results/transformer_baseline_fold10/` | 正式主模型結果 | 已確認 | `src.train`、`src.evaluate` | test accuracy 0.6547，macro F1 0.6644 |
| CNN augmentation smoke result | `results/cnn_aug_smoke_fold10/` | 驗證完整 augmentation training pipeline | 已確認 | `src.train` | 256/128/128 子集、1 epoch，非正式分數且不提交 |
| CNN augmentation fold 10 ablation | Google Drive controlled-search backup | 選擇 augmentation profile | 已確認 | validation Macro F1 | 已由 controlled search 的 initial phase 完成 |
| CNN controlled augmentation search | `docs/experiments/2026-08-13-cnn-controlled-augmentation-search.md` | 記錄四組初始比較、九輪單一變因迭代與唯一一次 test | 已確認 | Colab results + Drive backup | 最佳 validation Macro F1 0.7924；fold 10 test Macro F1 0.8536 |
| CNN EMA validation comparison | Google Drive EMA backup | 比較同次訓練的 online 與 EMA 權重 | 已確認 | `validation_metrics.json` | EMA F1 0.76515、online F1 0.76426；不採用 EMA |
| CNN 3-seed ensemble | `docs/experiments/2026-08-13-cnn-seed-ensemble.md` | 評估固定 seeds 42、123、2026 的 probability ensemble | 已確認 | Colab results + Drive backup | validation F1 0.7699；test F1 0.8501；未超越鎖定單一 CNN，故不採用 |
| CNN breakthrough protocol | `docs/experiments/2026-08-13-cnn-breakthrough-protocol.md` | 記錄 0.90 攻擊目標、多 fold validation 與五個候選 | 已確認 | 文件與 protocol tests | GPU 結果待補；未改善不合併回 main |
| 圖表目錄 | `figures/` | 儲存論文圖表 | 進行中 | 程式產生 | 已有 smoke confusion matrix，正式圖表待補 |

## 文獻與引用

| 成果 | 路徑 | 用途 | 狀態 | 證據來源 | 備註 |
| --- | --- | --- | --- | --- | --- |
| 文獻筆記 | `references/literature_notes.md` | 逐篇文獻整理 | 待補 | 文獻來源 | 需核心文獻 |
| 引用追蹤表 | `references/citation_tracker.md` | 追蹤章節引用需求 | 草稿 | 論文章節 | 需引用格式 |
| BibTeX 檔 | `references/references.bib` | 正式引用資料 | 待補 | 文獻來源 | 需匯入文獻 |

## 管理文件

| 成果 | 路徑 | 用途 | 狀態 | 備註 |
| --- | --- | --- | --- | --- |
| 專案儀表板 | `docs/dashboard.md` | 快速查看階段、風險與下一步 | 已確認 | 恢復工作時優先讀取 |
| 目前狀態 | `docs/current_status.md` | 接續工作前讀取 | 已確認 | 需隨研究資訊更新 |
| MVP 進度追蹤 | `docs/progress_tracker.md` | 追蹤已做、未做、結果與下一步 | 已確認 | 每次重要更新時同步維護 |
| AI 使用紀錄 | `docs/ai_usage_log.md` | 記錄 AI 協助範圍 | 草稿 | 需持續更新 |
| AI 揭露草稿 | `docs/ai_disclosure_draft.md` | 預備揭露文字 | 待確認 | 需教授與學校確認 |
| 第一週啟動計畫 | `docs/first_week_plan.md` | 啟動任務 | 已確認 | 等待使用者填資訊 |
| 教授問題清單 | `docs/professor_questions.md` | 準備會議問題 | 已確認 | 會後需更新會議紀錄 |
| 教授溝通模板 | `docs/professor_update_template.md` | 準備更新、詢問、會前與會後訊息 | 已確認 | 送出前需依情境調整 |
| 風險登記表 | `docs/risk_register.md` | 追蹤風險與阻塞 | 草稿 | 需每週檢查 |
| 里程碑 | `docs/milestones.md` | 追蹤進度 | 草稿 | 需日期與任務 |
| 研究日誌 | `docs/research_log.md` | 追蹤脈絡 | 草稿 | 需持續更新 |
| 決策紀錄 | `docs/decision_log.md` | 記錄重要決策 | 草稿 | 已有初始化決策 |

## 下一次應新增的成果

取得使用者或教授資訊後，優先新增或更新：

1. `docs/thesis_plan.md` 的研究方向與問題。
2. `docs/chapters/01_introduction.md` 的緒論初稿。
3. `docs/glossary.md` 的研究領域核心術語。
4. `docs/code_task_spec.md` 的第一個程式任務規格。
5. `docs/environment.md` 的程式語言與執行方式。
6. `src/` 或 `notebooks/` 的第一個可驗證程式原型。
7. `references/literature_notes.md` 的第一批核心文獻。
