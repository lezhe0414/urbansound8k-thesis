# 交付與提交前檢查清單

這份文件用於交給教授、階段性回報、論文初稿或程式成果提交前檢查。它不取代學校格式規範，而是確保本專案內的成果有足夠依據。

## 使用時機

- 準備交研究計畫給教授前。
- 準備交某章草稿前。
- 準備展示程式原型前。
- 準備提交實驗結果前。
- 準備整合論文初稿前。

## 基本檢查

- [ ] 已閱讀 `docs/current_status.md`。
- [ ] 已確認本次要交付的內容範圍。
- [ ] 已在 `docs/artifact_index.md` 找到對應成果。
- [ ] 已檢查 `docs/risk_register.md`，沒有未處理的高風險交付阻塞。
- [ ] 已更新 `docs/research_log.md`。
- [ ] 若涉及教授要求，已更新 `docs/meeting_notes.md`。
- [ ] 若涉及重要選擇，已更新 `docs/decision_log.md`。
- [ ] 若 AI Agent 協助了重要內容，已更新 `docs/ai_usage_log.md`。

## 論文內容檢查

- [ ] 本次交付章節放在 `docs/chapters/` 或指定位置。
- [ ] 章節標題符合目前論文大綱。
- [ ] 語氣符合 `docs/writing_style_guide.md`。
- [ ] 重要術語與縮寫已同步到 `docs/glossary.md`。
- [ ] 重要主張有文獻、資料或實驗支撐。
- [ ] 尚未有來源的句子標示 `[citation needed]`。
- [ ] 內容沒有把未驗證推測寫成已證實事實。
- [ ] 圖表、表格與結果能追溯到來源。

## 文獻與引用檢查

- [ ] 文獻已整理到 `references/literature_notes.md`。
- [ ] 待補引用已記到 `references/citation_tracker.md`。
- [ ] 引用格式符合教授或學校要求，若尚未確認則標示待確認。
- [ ] 參考文獻清單與正文引用一致。

## 程式碼檢查

- [ ] 程式碼位於 `src/` 或探索性 notebook 位於 `notebooks/`。
- [ ] 已記錄程式目的、輸入、輸出與執行方式。
- [ ] 已更新 `docs/environment.md`。
- [ ] 有最小可驗證方式。
- [ ] 程式輸出沒有覆蓋原始資料。
- [ ] 若使用環境變數，實際值不在 Git 中，只保留 `.env.example`。

## 資料與實驗檢查

- [ ] 原始資料放在 `data/raw/` 或已記錄外部位置。
- [ ] 處理後資料放在 `data/processed/` 或已記錄產生方式。
- [ ] 實驗紀錄使用 `docs/experiment_template.md` 或放在 `results/`。
- [ ] 結果檔、圖表與 log 放在 `results/` 或 `figures/`。
- [ ] 評估指標、參數與資料版本已記錄。

## 給教授前檢查

- [ ] 已整理本次交付摘要。
- [ ] 已列出希望教授確認的問題。
- [ ] 已列出下一步計畫。
- [ ] 已把教授可能會問的資料、方法、程式、引用來源準備好。
- [ ] 若需揭露 AI 使用，已檢查 `docs/ai_disclosure_draft.md`。

## Git 檢查

- [ ] 工作區沒有不明變更。
- [ ] 已執行 `python3 scripts/check_project_status.py` 或確認其檢查結果不影響本次交付。
- [ ] 重要變更已提交。
- [ ] 提交訊息能說明本次變更目的。
- [ ] 未提交大型資料、私人金鑰或敏感資料。
