# 第一週啟動計畫

這份文件用來把論文專案從「基礎設施已建立」推進到「可以開始寫作與寫程式」。

## 本週目標

本週最重要的目標不是直接寫完整論文，而是補齊能開始工作的最低資訊：

1. 確認論文大方向。
2. 確認教授近期要求。
3. 確認第一個程式任務。
4. 確認資料來源或資料樣本。
5. 確認近期截止日期。

## 第 1 步：補齊最短輸入

請先填以下三句，或直接貼到聊天裡：

```text
我的論文大方向是：
程式需要做的是：
教授最近要求我完成的是：
```

完成後，AI Agent 應更新：

- `docs/intake_questions.md`
- `docs/thesis_plan.md`
- `docs/current_status.md`
- `docs/milestones.md`

## 第 2 步：整理教授要求

將教授已經說過的要求整理到：

- `docs/meeting_notes.md`
- `docs/decision_log.md`
- `docs/task_inbox.md`

若教授要求還不清楚，使用 `docs/professor_questions.md` 準備下一次詢問。

## 第 3 步：建立第一版研究計畫

第一版研究計畫不需要完美，但應至少包含：

- 論文暫定題目
- 研究背景
- 研究問題
- 預期方法
- 程式碼需求
- 資料來源
- 預期成果

主要更新位置：

- `docs/thesis_plan.md`
- `docs/thesis_outline.md`
- `docs/chapters/01_introduction.md`

## 第 4 步：確認第一個程式任務

第一個程式任務應該小而可驗證，例如：

- 讀取資料樣本。
- 清理資料。
- 建立 baseline。
- 產生第一張圖。
- 實作核心方法的最小原型。

主要更新位置：

- `docs/environment.md`
- `src/README.md`
- `notebooks/README.md`
- `docs/experiment_template.md`

## 第 5 步：建立第一週檢查紀錄

每週結束前，更新：

- `docs/weekly_review.md`
- `docs/research_log.md`
- `docs/milestones.md`

## 第一週完成標準

- [ ] 已補上論文大方向。
- [ ] 已補上教授近期要求。
- [ ] 已確認第一個程式任務。
- [ ] 已確認資料來源或資料樣本。
- [ ] 已更新 `docs/thesis_plan.md`。
- [ ] 已更新 `docs/current_status.md`。
- [ ] 已建立下一週優先任務。

## 目前尚待使用者提供

```text
我的論文大方向是：
程式需要做的是：
教授最近要求我完成的是：
```
