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
