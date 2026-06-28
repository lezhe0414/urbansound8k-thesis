# Scripts

這個資料夾放專案輔助腳本。正式研究或實驗程式仍應依任務性質放在 `src/`，探索性分析可放在 `notebooks/`。

## 目前腳本

### `check_project_status.py`

檢查論文專案關鍵文件是否存在，並統計常見待補標記。

執行方式：

```text
python3 scripts/check_project_status.py
```

用途：

- 快速確認專案骨架是否完整。
- 找出仍有大量「待填」或「待確認」的文件。
- 提醒下一步應補論文方向、程式需求與教授要求。
