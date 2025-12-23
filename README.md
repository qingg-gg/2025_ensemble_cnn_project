## Snapshot Ensemble CNN Adversarial Defense (CIFAR-10)

這個專案使用 ResNet-18 / ResNet-34，在 CIFAR-10 上實作 Snapshot Ensemble，
並透過 FGSM 產生對抗樣本，評估模型在不同程度攻擊下的魯棒性。

### 功能
- ResNet-18 / ResNet-34 訓練與評估
- Snapshot Ensemble（多個模型快照集成）
- FGSM 對抗攻擊
- 輸出結果 json 與圖表

### 資料夾結構
```
├─ src/                     主程式
│ ├─ cnn.py                 訓練與評估
│ ├─ quick_test.py          快速測試
│ └─ visualize_result.py    結果視覺化
├─ data/                    訓練資料
├─ snapshots/               模型權重
└─ results/                 結果
```

### 快速開始
1. 建立虛擬環境
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```
2. 安裝套件
   ```
   pip install -r requirements.txt
   ```
3. 快速測試
   ```
   python src/quick_test.py
   ```
4. 視覺化成果
    ```
   python src/visualize_result.py --result_file ./results/result.json
   ```

### 注意事項
- result 資料夾中有輸出影像範例
- 執行 cnn.py、quick_test.py、visualize_result.py 之前，務必確定 main() 中的各項參數，或是在 Terminal 執行時補齊

### License
本專案僅供課程作業使用。