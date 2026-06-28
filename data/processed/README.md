# 處理後資料

這裡放由程式產生或清理後的資料。

## 命名建議

使用能表達來源與日期的檔名，例如：

```text
cleaned_dataset_2026-06-28.csv
features_v1.parquet
annotated_sample_round1.json
```

## 每份資料應能追溯

- 來自哪份原始資料；
- 用哪個程式產生；
- 使用哪些參數；
- 用於哪個實驗；
- 是否能重複產生。

## 注意

`.gitignore` 預設會忽略此資料夾中的資料檔，只保留 README 與 `.gitkeep`。重要處理流程應放在 `src/`，不要只保留輸出檔。
