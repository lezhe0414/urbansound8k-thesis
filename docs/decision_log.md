# 決策紀錄

這份文件用來記錄論文與程式碼開發中的重要決策。每個決策都應該能說明「為什麼這樣做」，避免之後忘記原因或重複討論。

## 使用時機

當發生以下情況時，請新增一筆決策：

- 確定或更改論文題目。
- 確定研究問題、研究方法或評估方式。
- 決定使用某個資料集、模型、演算法、框架或程式語言。
- 教授要求改方向。
- 實驗結果顯示原本方法不可行，需要調整。
- 決定論文章節架構或投稿 / 繳交格式。

## 決策模板

### DEC-000：決策標題

- 日期：
- 狀態：提議 / 已決定 / 已取代
- 相關文件：
- 相關會議：

#### 背景

-

#### 選項

1. 
2. 
3. 

#### 決策

-

#### 理由

-

#### 影響

- 對論文：
- 對程式：
- 對資料：
- 對時程：

#### 後續行動

- [ ] 
- [ ] 

---

## DEC-001：建立 AI Agent 論文協作專案

- 日期：2026-06-28
- 狀態：已決定
- 相關文件：`README.md`、`AGENTS.md`
- 相關會議：`docs/meeting_notes.md`

#### 背景

使用者需要一個專案來協助完成論文寫作、程式碼開發、資料整理、實驗紀錄與教授回饋追蹤。教授已同意使用 AI Agent。

#### 決策

建立以文件、程式碼、資料、實驗結果與圖表為核心的專案骨架，並用 `AGENTS.md` 定義 AI Agent 協作規則。

#### 理由

論文和程式碼需要長期迭代，必須讓研究問題、教授要求、程式環境、實驗結果和寫作草稿能被追蹤。

#### 影響

- 對論文：後續章節、研究日誌、文獻筆記會有固定位置。
- 對程式：正式程式放在 `src/`，探索性分析放在 `notebooks/`。
- 對資料：原始資料、處理後資料、結果與圖表分開管理。
- 對時程：使用 `docs/milestones.md` 追蹤近期任務。

#### 後續行動

- [x] 補上論文題目或暫定方向。
- [ ] 補上教授近期要求。
- [x] 決定第一個要實作的程式任務。

---

## DEC-002：保留 CNN baseline，將 Transformer 作為延伸比較

- 日期：2026-06-28
- 狀態：提議
- 相關文件：`docs/thesis_plan.md`、`docs/code_task_spec.md`
- 相關會議：待教授確認

#### 背景

Project definition 明確寫到使用 CNN 對 spectrogram representations 進行 sound event classification。使用者提出是否可改用 Transformer，因為可能比 CNN 更好。

#### 選項

1. 完全照 definition，只做 CNN。
2. 完全改成 Transformer。
3. 先完成 CNN baseline，再把 Transformer 作為比較或延伸模型。

#### 決策

採用選項 3：先完成 CNN baseline，之後若時間允許再加入 Transformer 或 transfer learning model 比較。

#### 理由

CNN 是 definition 明確承諾的交付，先完成可降低偏離題目與進度風險。Transformer 可能在預訓練或足夠資料下有較好表現，但從零訓練需要更多資料、時間與算力。將 Transformer 作為比較模型，可以保留創新性，同時不破壞原本研究方向。

#### 影響

- 對論文：論文主線仍是 spectrogram-based sound event detection，結果章可比較 CNN 與額外模型。
- 對程式：第一階段只實作資料處理、CNN training、evaluation；第二階段再加 Transformer。
- 對資料：先選一個資料集完成流程。
- 對時程：可追回落後進度，避免一開始就做過大範圍。

#### 後續行動

- [ ] 完成 CNN baseline pipeline。
- [ ] 向教授確認 Transformer comparison 是否合適。
- [ ] 若 baseline 已有結果，再新增 Transformer 實驗。

---

## DEC-003：以 validation-only ablation 選擇 CNN 資料增強策略

- 日期：2026-08-13
- 狀態：已決定
- 相關文件：`src/data/augmentation.py`、`configs/cnn_aug_*.yaml`、`scripts/run_cnn_augmentation_ablation.py`
- 相關會議：2026-08-07 supervisor meeting

#### 背景

CNN fold 10 evaluation 已達約 0.81--0.83 accuracy 與 0.82--0.84 Macro F1，繼續只調 learning rate 或 dropout 的改善有限。需要測試能否透過資料增強降低 overfitting，同時避免用 test 結果反覆選參數。

#### 決策

固定 CNN 架構、training seed、optimizer、class weighting、class-aware sampling 與 epoch，建立 control、light、balanced、strong 四組 augmentation profile。使用 validation Macro F1 選擇最佳 profile，然後只對勝出設定執行 fold 10 test evaluation。

#### 理由

此設計可將差異主要歸因於 augmentation 強度，也避免 test leakage。逐樣本增強比原本整個 batch 共用一次遮罩決策更具多樣性；Mixup/CutMix 使用逐樣本 mixing coefficient，使 loss 與訓練指標更一致。

#### 影響

- 對論文：可加入一個有控制組的 augmentation ablation，而非只陳述反覆調參。
- 對程式：新增即時 spectrogram augmentation，不必重新 preprocessing。
- 對資料：raw 與 processed data 均不修改。
- 對時程：先跑 fold 10 選定策略，再只對勝出設定做 seed stability 與正式 10-fold。

#### 後續行動

- [ ] 在 Colab 跑四組 fold 10 validation ablation。
- [ ] 保存 validation 排名與勝出設定的 test evaluation artifacts。
- [ ] 根據結果建立單一 final config，再做 seed ensemble 或 10-fold。

#### 受控迭代補充

正式執行採用最多十輪的 greedy controlled search。每輪從目前 validation Macro F1 最佳設定複製，只調整一類變因；改善才保留，否則退回。連續五輪未改善即停止。每組 run 使用唯一名稱並立即記錄及備份，fold 10 test 在唯一設定鎖定前保持不可見，Accuracy 僅作輔助說明。

---

## DEC-004：鎖定 CNN Mixup 設定並轉入 10-fold 最終驗證

- 日期：2026-08-13
- 狀態：已決定
- 相關文件：`docs/experiments/2026-08-13-cnn-controlled-augmentation-search.md`
- 相關會議：2026-08-07 supervisor meeting

#### 背景

受控搜尋完成四組初始比較與九輪單一變因迭代。Strong profile 為初始勝出設定；將其 frequency mask 調整為 9 並將 Mixup probability 調整為 0.45 後，validation Macro F1 由 0.7800 提升至 0.7924。後續五輪均未改善。

#### 決策

鎖定 `cnn_aug_cnn_aug_20260813_1903_iter04_mixup_probability` 為 CNN 最終候選設定，不再根據 fold 10 test 或後續 fold 結果調參。下一階段僅用固定設定執行正式 10-fold cross-validation。

#### 理由

選模完全依 validation Macro F1，符合預先定義的停止條件，且唯一一次 fold 10 test Macro F1 為 0.8536，較歷史結果 0.8413 高約 0.0123。單一 fold 尚不足以證明可泛化提升，因此需要 10-fold mean 與 standard deviation。

#### 影響

- 對論文：可誠實報告受控搜尋方法、有效與無效變因，以及單一 fold 的限制。
- 對程式：保留現有 runner 與設定，不再增加 fold 10 調參輪次。
- 對資料：原始音訊與 `.npz` cache 均未修改。
- 對時程：下一個主要算力工作為固定設定的 10-fold cross-validation。

#### 後續行動

- [x] 將唯一最佳 resolved config 保存為可重複執行的 final config。
- [ ] 以固定 final config 執行 10-fold cross-validation。
- [ ] 整理 Macro F1、Accuracy mean/std 與 aggregate confusion matrix。

---

## DEC-005：在 10-fold 前進行一次 validation-only EMA 比較

- 日期：2026-08-13
- 狀態：已決定
- 相關文件：`src/utils/ema.py`、`configs/cnn_aug_ema.yaml`
- 相關會議：無；使用者後續實驗決策

#### 背景

受控 augmentation 搜尋已降低模型在增強訓練指標與 validation 指標之間的差距。使用者希望在算力有限的前提下繼續提高 Macro F1，並優先嘗試比 3-seed ensemble 成本低的 EMA 權重。

#### 決策

保留 `configs/cnn_aug_final.yaml` 不變，新增 validation-only EMA 候選設定。一次訓練同時保留 online 權重及其 EMA 副本，每個 epoch 同時評估兩者；`val_f1_macro` 代表 EMA 並保存至 `best_model.pt`，最佳 online Macro F1 另存至 `best_online_model.pt`。EMA 候選禁止執行 fold 10 test。實驗完成後，EMA 最佳 Macro F1 為 0.76515，online 最佳 Macro F1 為 0.76426，差異僅 0.00089，因此正式設定關閉 EMA。

#### 理由

EMA 平滑 stochastic augmentation、Mixup 與 mini-batch 更新造成的短期參數波動，只增加一套權重與一次 validation forward pass，不需要三次完整訓練。同次訓練比較可避免把不同 seed 的自然波動誤認為 EMA 效果。候選 decay 固定為 `0.995`；以每 epoch 約 220 個 updates 計算，其有效平滑範圍約涵蓋最近 200 次更新，較適合目前只有 10 epochs 的短訓練，避免把過早期、尚未收斂的權重保留太久。

#### 影響

- 對論文：若 EMA 有效，需清楚報告 decay、選模指標及額外 validation 成本。
- 對程式：checkpoint 保持 `model_state` 相容，並標記 `checkpoint_source`；EMA run 額外保存 online 權重及指標。
- 對評估：不得再次使用已看過的 fold 10 test；EMA 決策只依 validation Macro F1。
- 對時程：EMA 未帶來實質改善，後續已改執行固定 3-seed ensemble 作穩定性檢查。

#### 後續行動

- [x] 在 Colab 執行 `configs/cnn_aug_ema.yaml` validation-only run。
- [x] 比較同次 run 的 online 與 EMA validation Macro F1。
- [x] 鎖定 online 權重策略，正式設定關閉 EMA。
- [x] 以固定三個 seed 平均預測機率，不挑選單一最佳 seed。

---

## DEC-006：不採用 3-seed ensemble 作為主要 CNN 設定

- 日期：2026-08-13
- 狀態：已決定
- 相關文件：`src/ensemble.py`、`configs/cnn_aug_final.yaml`、`docs/experiments/2026-08-13-cnn-seed-ensemble.md`
- 相關會議：無；使用者後續實驗決策

#### 背景

EMA 的 validation 改善僅 0.00089，因此改以 seeds 42、123、2026 訓練三個相同 CNN。各模型只依 validation Macro F1 保存最佳 checkpoint，ensemble 使用三者 softmax probability 的算術平均，EMA 與個別 seed test 均關閉。

#### 決策

3-seed ensemble 實驗完成但不作為主要設定。正式 10-fold 仍使用已鎖定的單一 CNN `configs/cnn_aug_final.yaml`，且 EMA 關閉。

#### 理由

Ensemble validation Macro F1 為 0.7699，低於鎖定單一 CNN 的 0.7924，也低於最佳單一 seed 42 的 0.7868。此 validation 結果已足以否決採用 ensemble。鎖定後唯一一次 fold 10 test Macro F1 為 0.8501，也略低於單一 CNN 的 0.8536，但 test 結果只作確認，不作選模依據。

#### 影響

- 對論文：可將 ensemble 列為沒有改善的穩定性實驗，避免只呈現正面結果。
- 對程式：保留 `src/ensemble.py` 供重現，不刪除已驗證功能。
- 對評估：不再根據 fold 10 調整 ensemble 權重、seed 或超參數。
- 對時程：算力集中到固定單一 CNN 的正式 10-fold cross-validation。

#### 後續行動

- [x] 完成三個固定 seeds 的 validation-only training。
- [x] 完成 validation probability ensemble 比較。
- [x] 鎖定後執行唯一一次 ensemble fold 10 test。
- [ ] 使用固定單一 CNN 設定執行正式 10-fold cross-validation。

---

## DEC-007：以鎖定單一 CNN 執行正式 UrbanSound8K 10-fold

- 日期：2026-08-17
- 狀態：已預註冊；待執行
- 相關文件：`configs/cnn_aug_final.yaml`、`docs/experiments/2026-08-17-cnn-formal-10fold.md`
- 相關會議：延續 2026-08-07 supervisor meeting 與受控搜尋決策

#### 背景

CNN controlled search、EMA、固定三 seed ensemble 與高風險 breakthrough study 均已
完成。沒有延伸方法在 folds 1、4、7 顯示足以取代鎖定設定的穩健改善，因此繼續搜尋
會增加 development overfitting 風險。

#### 決策

使用 `configs/cnn_aug_final.yaml` 的固定 hash、seed 42、EMA off 與單一模型，依
UrbanSound8K 十個官方 folds 執行正式 cross-validation。每個 test fold 對應固定的下一
fold 作 validation；checkpoint 只依 validation Macro F1 選擇，test fold 各評估一次。

#### 理由

正式 10-fold mean/std 比單一 fold 高分更能回答方法是否跨錄音條件泛化，也能量化
fold variability。固定設定可避免在已觀察結果後改變方法，保留研究可信度。

#### 影響

- 對論文：此結果將成為 from-scratch CNN 的主要正式數據。
- 對程式：新增可恢復但禁止重複 test 的 runner、predictions 與 aggregate summary。
- 對資料：raw audio 與既有 Mel cache 唯讀，不重新 preprocessing。
- 對時程：完成後停止 from-scratch CNN 調參，轉入三模型比較與論文定稿。

#### 後續行動

- [ ] 在固定 commit 完成十個 folds。
- [ ] 核對每 fold completion manifest 與 Drive backup。
- [ ] 整理 mean/std、per-class F1 及 aggregate confusion matrix。
- [ ] 更新 Results、Discussion、limitations 與 PDF。
