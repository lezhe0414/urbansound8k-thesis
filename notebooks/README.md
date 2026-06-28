# Notebook 目錄

`notebooks/` 用來放探索性分析、資料檢查、快速原型與暫時性實驗。

## 使用原則

- Notebook 可以用來快速嘗試想法，但重要邏輯最後應整理到 `src/`。
- 每個 notebook 檔名應包含日期或任務，例如 `2026-06-28-data-check.ipynb`。
- Notebook 中產生的重要結果，應匯出到 `results/` 或 `figures/`。
- 如果 notebook 支撐論文主張，請在 `docs/research_log.md` 記錄其用途與結論。

## 建議 notebook 開頭包含

- 目的：
- 對應研究問題：
- 使用資料：
- 主要輸出：
- 結論摘要：

## 注意

不要讓 notebook 成為唯一可重現的實驗來源。最後論文需要依賴的流程，應整理為正式腳本或清楚的執行步驟。
