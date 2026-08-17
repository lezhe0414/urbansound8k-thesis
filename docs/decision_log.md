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

## DEC-007：以 EfficientAT MN10 建立第三個 transfer-learning 比較模型

- 日期：2026-08-17
- 狀態：已決定
- 相關文件：`docs/experiments/pretrained-cnn-transfer.md`、`configs/pretrained_cnn_linear_probe.yaml`
- 相關會議：2026-08-07 supervisor meeting 後續研究規劃

#### 背景

From-scratch CNN 已是主要基準，從零訓練 Spectrogram Transformer 則受限於 UrbanSound8K 的資料量。需要加入一個 AudioSet-pretrained CNN，評估 transfer learning 是否能在不改變主要 CNN 研究主線的情況下提供較穩健的 Macro F1。

#### 決策

採用 MIT-licensed EfficientAT `mn10_as` 作為第三個模型。先在 folds 1、4、7 做五 epochs linear probing；只在平均 validation Macro F1 接近 control `0.7818` 或有清楚上升趨勢時，才從最佳 linear-probe checkpoint 接續 partial fine-tuning。Fold 10 保持封存。

#### 理由

`mn10_as` 是 AudioSet-pretrained MobileNetV3-style CNN，官方規模約 4.88M parameters、0.54 GMACs，較大型 PANNs CNN14 或 AST 更符合目前時間及 Colab 運算限制。官方 frontend 接收 32 kHz waveform，可避免把本專案既有、逐 clip 標準化的 Mel cache 錯誤重用於不同預訓練分布。

#### 影響

- 對論文：新增 transfer-learning 比較，但 CNN 仍是主要模型。
- 對程式：新增 raw-waveform dataset、獨立 cache、EfficientAT wrapper 及 development runner。
- 對資料：不覆寫 raw audio 或既有 Mel cache；新 cache 使用獨立路徑。
- 對評估：唯一選模指標為三-fold validation Macro F1 mean/std，fold 10 不參與。

#### 後續行動

- [x] 在 Colab A100 完成 folds 1、4、7 linear probing。
- [x] 依預先定義門檻決定並完成 partial fine-tuning。
- [x] 鎖定唯一設定後執行一次 fold 10 test evaluation。

---

## DEC-008：鎖定 EfficientAT encoder LR 2e-5 並完成唯一 fold 10 final evaluation

- 日期：2026-08-17
- 狀態：已決定
- 相關文件：`configs/pretrained_cnn_partial_finetune_lr2e5.yaml`、`configs/pretrained_cnn_partial_finetune_final_test.yaml`、`docs/experiments/pretrained-cnn-transfer.md`
- 相關會議：2026-08-07 supervisor meeting 後續研究規劃

#### 背景

EfficientAT linear probing 在 folds 1、4、7 達到 Macro F1 `0.8471 ± 0.0327`，高於相同 development protocol 的 from-scratch CNN control `0.7818`。Partial fine-tuning v1 使用 encoder/head learning rates `1e-5`/`3e-4`，提高至 `0.8688 ± 0.0280`。依 protocol 只允許再測一個鄰近設定。

#### 決策

鄰近設定只將 encoder learning rate 改為 `2e-5`，其餘條件固定，得到 development Macro F1 `0.8716 ± 0.0283` 及 Accuracy `0.8734 ± 0.0212`。因其 Macro F1 比 v1 高 `0.0028`，故只依 development 結果鎖定此設定。鎖定後以預先選定的 validation fold 4 執行一次 fold 10 test，Macro F1 `0.9041`、Accuracy `0.8949`；不再建立或評估其他 test candidates。

#### 理由

選模遵循預先定義的唯一主要指標，且 fold 10 在設定鎖定前完全封存。鄰近設定的改善幅度雖小，但三個 folds 的平均值均依相同 seed、epochs、資料切分及評估方式取得。一次性 test 只用於 final confirmation，未回饋到設定選擇。

#### 影響

- 對論文：EfficientAT 是 transfer-learning 第三模型，結果包含 AudioSet 預訓練的貢獻，不能描述成純架構公平比較。
- 對評估：單一 fold 10 Macro F1 `0.9041` 不是 10-fold 泛化證明；fold 間變異仍需正式 10-fold 檢驗。
- 對模型：from-scratch CNN 保留為主要基準，EfficientAT 是目前表現較高的 transfer-learning comparison。
- 對時程：停止 pretrained CNN 超參數搜尋；下一步只執行固定設定的正式 cross-validation。

#### 後續行動

- [x] 將全部 development 與 final artifacts 備份到 Google Drive。
- [x] 確認 fold 10 只評估一個鎖定設定。
- [ ] 視時程執行固定 EfficientAT 設定的正式 10-fold cross-validation。
- [ ] 在論文加入三模型公平比較與限制分析。

---

## DEC-009：鎖定 EfficientAT v2 weak Mixup 與 last-3-block fine-tuning

- 日期：2026-08-17
- 狀態：已決定
- 相關文件：`configs/pretrained_cnn_v2_mixup_last3.yaml`、`docs/experiments/pretrained-cnn-transfer-v2.md`
- 相關會議：無；使用者要求的 development-only 延伸實驗

#### 背景

EfficientAT v1 在 folds 1、4、7 的 development Macro F1 為 `0.8716 ± 0.0283`。為判斷五 epochs 是否尚未收斂，並測試 gradual unfreezing、pretrained-specific augmentation 與解凍深度，建立 v2 受控序列。所有選擇只依三-fold validation Macro F1；v1 fold 10 test 不參與，v2 不再執行 fold 10。

#### 決策

鎖定 8 epochs、post-frontend Mixup (`alpha=0.15`, `probability=0.5`)、解凍最後 3 個 convolution blocks、encoder/head learning rates `2e-5`/`3e-4` 的方法。唯一設定檔為 `configs/pretrained_cnn_v2_mixup_last3.yaml`。後續 EfficientAT 正式 10-fold 應使用這組固定方法參數，不再根據單一 development fold 調整。

#### 理由

此設定在 folds 1、4、7 達到 Macro F1 `0.8844 ± 0.0165` 與 Accuracy `0.8824 ± 0.0141`，相較 v1 提高 `0.0128` 且 fold 間標準差降低。它也比 last-2-block Mixup 高 `0.0057`，主要改善較弱的 fold 7，而 folds 1、4 沒有明顯退化。Gradual unfreezing 與輕量 time/frequency masking 均未穩健超越 8-epoch control。

#### 影響

- 對模型：v2 固定方法取代 v1 作為未來 EfficientAT cross-validation 的候選，但不改寫 v1 已完成的一次性 fold 10 結果。
- 對評估：v2 尚無 test result；不得把 v1 fold 10 Macro F1 `0.9041` 標示為 v2 表現。
- 對論文：報告三-fold mean/std、額外 AudioSet pretraining 及 validation-only 選模流程，不能把差異歸因於純架構。
- 對程式：正式 10-fold 前需新增 fixed-config split support，確保每個 test fold 不參與 validation selection。

#### 後續行動

- [x] 完成 A-D 受控比較與 last 1/2/3 depth study。
- [x] 核對六個 runs 的 Drive summaries、manifests 與 checkpoints。
- [x] 確認所有 v2 runs 均為 `test_evaluated=false`。
- [ ] 以鎖定方法執行正式 10-fold cross-validation。
- [ ] 將 mean/std、per-class F1 與 aggregate confusion matrix納入論文。

---

## DEC-010：以 development-only recommended study 決定唯一正式 10-fold 方法

- 日期：2026-08-17
- 狀態：已完成
- 相關文件：`docs/experiments/pretrained-cnn-recommended-study.md`、`configs/pretrained_cnn_mn20_locked_last2.yaml`
- 相關會議：無；使用者要求的 pretrained CNN 延伸與正式驗證

#### 背景

EfficientAT MN10 v2 在 folds 1、4、7 達到 Macro F1 `0.8844 ± 0.0165`，但仍未完成正式 10-fold。單純反覆調整 learning rate、dropout 或解凍深度的邊際效益已下降，因此後續集中於資料不變性、推論平均、固定 seed ensemble 與較寬的 AudioSet-pretrained CNN。

#### 決策

已在完全封存 fold 10 的 folds 1、4、7 比較動態 waveform augmentation、zero-fill time-shift TTA、固定 seeds 42/123/2026 probability ensemble 與 EfficientAT MN20。只依 mean validation Macro F1 鎖定 `configs/pretrained_cnn_mn20_locked_last2.yaml`：MN20、seed 42、解凍最後 2 blocks、無 waveform augmentation、無 TTA。正式 10-fold 使用預先固定 cyclic validation mapping，每個 test fold 只評估一次。

#### 理由

勝出設定達 validation Macro F1 `0.89069 ± 0.01165`，高於 MN10 v2 control `0.88437 ± 0.01650`，mean 增加 `0.00631` 且 fold variance 降低。MN10 三 seed ensemble 只增加 `0.00026` 且 variance 惡化；三種 waveform augmentation 與 TTA 均降低 mean Macro F1。這個選擇完全限制在 development folds，沒有使用歷史或新的 fold 10 結果回饋調參。

#### 影響

- 對資料：augmentation 只在 cache 載入後動態作用，不修改 raw audio 或 cache。
- 對模型：MN20 與 MN10 使用相同官方 frontend，差異主要是 pretrained backbone 容量。
- 對算力：若三 seed ensemble 勝出，正式 10-fold 必須完整訓練三個固定 seeds，不能事後挑 seed。
- 對論文：必須分開報告 development selection 與 formal 10-fold results，並揭露 AudioSet pretraining。
- 對結果：唯一 MN20 方法的正式 Macro F1 為 `0.87686 ± 0.04048`、Accuracy 為 `0.86883 ± 0.04263`；不得再根據十個 test folds 改設定。

#### 後續行動

- [x] 完成三組 waveform augmentation development runs。
- [x] 完成 TTA、三 seed ensemble 與 MN20 development 比較。
- [x] 只依 mean validation Macro F1 鎖定唯一方法。
- [x] 執行一次固定 formal 10-fold 並備份 artifacts。
- [x] 核對十個 test predictions、十個 checkpoints、aggregate summary 與 Drive backup。

---

## DEC-011：正式 10-fold 後的 MN20 延伸僅作 development-only 探索

- 日期：2026-08-17
- 狀態：已決定
- 相關文件：`docs/experiments/pretrained-cnn-postformal-exploration.md`
- 相關會議：無；使用者要求延伸三 seed ensemble、checkpoint averaging 與 loss study

#### 背景

MN20 的固定正式 10-fold 已完成，其 test-fold 結果已被觀察，因此後續調整不能再被描述為原正式 protocol 的一部分，也不能依正式 fold 或類別表現作選擇。

#### 決策

建立獨立 branch，僅使用既有 development folds 1、4、7 的 mean validation Macro F1 比較：固定 seeds 42/123/2026 probability ensemble、epochs 5--8 的 validation top-3 checkpoint weight average，以及唯一 loss 變因 class-balanced focal loss (`gamma=1.5`)。Cross-entropy 與 focal loss 共用完全相同的 linear-probe checkpoints；所有 runner 禁止 test evaluation。

#### 理由

這三項方法分別處理 seed variance、訓練軌跡 variance 與困難樣本權重，而且不需要改變 waveform cache、官方 frontend、backbone 容量或資料切分。共用初始化及固定其餘參數可將差異歸因於單一方法。

實驗完成後，focal loss + 三 seed probability ensemble 以 Macro F1 `0.89395 ± 0.00932` 勝出，比鎖定 development control `0.89069 ± 0.01165` 高 `0.00327`，且標準差略降。Weighted-CE ensemble 為 `0.89267 ± 0.01052`。Checkpoint averaging 在 CE 與 focal 下均降低 mean F1，因此不保留。這些數字只來自 folds 1、4、7 validation。

#### 影響

- 對結果：新數字必須標示為 post-formal exploratory validation，不得取代 `0.87686 ± 0.04048` 的正式 10-fold 結果。
- 對資料：不重新 preprocess，不修改 raw audio 或 waveform/Mel cache。
- 對論文：可作為延伸實驗與 future work 證據，但不能宣稱是獨立 test-set 改善。
- 對 Git：程式、設定與文件提交到獨立分支；checkpoints/results 只備份到 Drive。

#### 後續行動

- [x] 在 Colab 完成全部 27 個 development-only runs。
- [x] 依 mean validation Macro F1 整理各方法效果與 variance。
- [x] 確認所有 27 份 manifests 的 `test_evaluated=false`。
- [x] 將結果、限制與 Drive 備份寫回實驗紀錄。
