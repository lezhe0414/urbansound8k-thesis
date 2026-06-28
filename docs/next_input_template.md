# 下一次給 AI Agent 的輸入範本

這份文件可以直接複製到聊天裡填寫。資訊不完整也沒關係，AI Agent 會根據已填內容更新論文計畫、章節、任務、程式或文獻文件。

## 最短版本

```text
我的論文大方向是：
程式需要做的是：
教授最近要求我完成的是：
```

## 建議版本

```text
我的論文大方向是：

教授目前同意或要求的研究範圍是：

教授最近要求我完成的是：

我需要寫的程式大概是：

目前有的資料是：

我希望下一步先產出：

近期截止日期是：
```

## 如果你要我幫你寫論文

```text
請幫我處理論文章節：

目前章節內容或想法：

教授意見：

需要的語氣：

引用或文獻來源：

希望輸出：
```

## 如果你要我幫你寫程式

```text
請幫我處理程式任務：

程式目的：

輸入資料：

輸出結果：

希望使用的語言或工具：

目前已有檔案：

驗證方式：
```

## 如果你要我整理文獻

```text
請幫我整理文獻：

文獻檔案或連結：

我想知道的重點：

要對應的論文章節：

是否需要比較表：

引用格式要求：
```

## AI Agent 收到後應做的事

1. 先讀 `docs/current_status.md`。
2. 更新 `docs/intake_questions.md` 或 `docs/task_inbox.md`。
3. 若資訊足夠，更新 `docs/thesis_plan.md`。
4. 若涉及寫作，更新 `docs/thesis_outline.md` 或 `docs/chapters/`。
5. 若涉及程式，更新 `docs/environment.md`，並在 `src/` 或 `notebooks/` 建立工作。
6. 若涉及文獻，更新 `references/literature_notes.md` 或 `references/citation_tracker.md`。
7. 將重要進度寫入 `docs/research_log.md`。
