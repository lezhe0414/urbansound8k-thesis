# Reports

這個資料夾放由輔助腳本產生的檢查報告。

## Project Status Report

產生方式：

```text
python3 scripts/check_project_status.py --write-report
```

預設輸出：

```text
reports/project_status.md
```

報告用途：

- 保存某次狀態檢查結果。
- 每週檢查時快速比較待補項目。
- 交付前確認關鍵文件是否存在。
