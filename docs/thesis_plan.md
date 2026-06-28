# 論文計畫

## 基本資訊

- 論文暫定題目：Sound Event Detection Using Machine Learning Techniques
- 學校 / 系所：Queen Mary University of London, School of Electronic Engineering and Computer Science
- 指導教授：Lin Wang
- 學位：MSc
- 預計完成日期：2026-08-19 final submission；之後準備 presentation video 與 viva

## 研究背景

聲音事件偵測需要在真實環境中辨識不同聲音類別，例如鳥聲、無人機聲與背景噪音。由於音訊訊號本身是一維時間序列，本專案將音訊轉換為 Mel-spectrogram 等二維表示，使問題能以 visual data analysis 的方式處理。

Project definition 明確指定以 CNN 進行 spectrogram-based audio classification。考量目前進度落後，論文應聚焦在可重複執行的資料處理、baseline CNN、實驗比較與結果分析；Transformer 可作為延伸比較，但不應在未確認前完全取代 CNN。

## 研究問題

1. How effectively can CNN-based models classify sound events from Mel-spectrogram representations?
2. How do different spectrogram representations or model configurations affect classification performance?
3. If time allows, can a Transformer-based or transfer learning model improve performance compared with the CNN baseline?

## 研究目標

- 建立公開音訊資料集的 preprocessing pipeline。
- 將音訊轉換為 spectrogram/Mel-spectrogram 表示。
- 實作並訓練 CNN baseline sound event classifier。
- 使用 accuracy、precision、recall、F1-score 與 confusion matrix 評估模型。
- 比較不同模型設定；若 baseline 完成，再加入 Transformer 作為延伸比較。

## 方法與技術路線

1. Dataset：優先選 UrbanSound8K 或 ESC-50，先使用一個資料集完成端到端流程。
2. Preprocessing：統一取樣率、音訊長度、normalisation，產生 Mel-spectrogram。
3. Baseline model：以 PyTorch 實作小型 CNN 或 transfer learning CNN。
4. Training：加入資料增強、regularisation 與基本 hyperparameter tuning。
5. Evaluation：輸出整體與逐類別 metrics，產生 confusion matrix。
6. Extension：若時間允許，加入 Transformer-based spectrogram classifier 作比較。

## 程式碼需求

- 主要程式語言：Python
- 需要的套件 / 框架：PyTorch、Librosa、NumPy、scikit-learn、Matplotlib
- 輸入資料：UrbanSound8K 或 ESC-50 audio files 與 label metadata
- 輸出結果：processed spectrograms、trained model weights、metrics CSV/JSON、confusion matrix figures
- 需要重複執行的實驗：CNN baseline、不同 spectrogram 設定、可選 Transformer/transfer learning 比較

## 預期成果

- 論文章節：introduction、literature review、methodology、experiments/results、discussion、conclusion
- 程式碼：資料處理、模型定義、訓練、評估、圖表輸出
- 實驗結果：baseline CNN metrics 與至少一組比較設定
- 圖表：pipeline diagram、Mel-spectrogram example、training curves、confusion matrix
- 其他產出：可重複執行說明、AI usage log、教授確認紀錄

## 風險與待釐清問題

- 進度已落後原本 6 月底 baseline CNN 初步結果的時程。
- Transformer 若完全取代 CNN，可能偏離 project definition；建議先做 CNN baseline，再將 Transformer 作為比較。
- 資料集尚未下載與驗證，需要優先處理。
- 8 頁 PDF 空間有限，模型數量不宜過多。

## 近期下一步

1. 選定資料集並建立資料目錄結構。
2. 寫出 Mel-spectrogram preprocessing。
3. 寫出 CNN baseline training/evaluation。
4. 用一小批資料驗證 pipeline 可跑通。
5. 向教授確認 Transformer 是否可作為額外比較模型。
