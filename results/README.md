# 實驗結果

`results/` 用來放實驗輸出、評估指標、表格、log 與中間結果。

## 建議結構

每次正式實驗建立一個資料夾：

```text
results/
└── 2026-06-28_experiment-name/
    ├── README.md
    ├── metrics.csv
    ├── output.json
    └── run.log
```

## 每次實驗應記錄

- 實驗編號：
- 日期：
- 目的：
- 對應研究問題：
- 對應程式：
- 對應資料：
- 執行命令：
- 參數：
- 評估指標：
- 結果摘要：
- 對論文的意義：

可使用 `docs/experiment_template.md` 作為紀錄格式。

## 注意

`.gitignore` 預設會忽略此資料夾中的輸出檔，只保留 README 與 `.gitkeep`。是否納入結果檔需依檔案大小與論文重現需求決定。
