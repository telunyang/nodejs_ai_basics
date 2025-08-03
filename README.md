# nodejs_ai_basics
使用 `nodejs` 開發 AI 基礎應用的範例程式碼。

## node.js 網址
- [node.js 官方網站](https://nodejs.org/)
- [NVM for Windows (上課會用到)](https://github.com/coreybutler/nvm-windows)
- [NVM for Mac/Linux](https://github.com/nvm-sh/nvm)

## 套件安裝
```bash
# 指定安裝
npm i @huggingface/transformers @google/genai canvas wavefile @lancedb/lancedb --save

# 或是批次安裝
npm i --save
```

## 大綱
- AI 基礎概論與演進
- AI 應用概論
  - 電腦視覺
    - 圖片特徵擷取與相似度計算
    - 物件偵測
    - 圖片分類
  - 自然語言處理
    - 文字特徵擷取與相似度計算
    - 文字分類
    - 情感分析
    - 文字生成
    - 翻譯
  - 語音處理
    - 音訊特徵擷取與相似度計算
    - 語音轉文字
    - 文字轉音樂
    - 文字轉語音
  - 知識挖掘
    - 建立向量資料庫
    - 向量檢索 (使用文字描述)
    - 向量檢索 (使用音訊檔案)
  - 智慧文件處理
    - 圖片內容檢視
    - 語音內容檢視
  - 生成式 AI (使用 Gemini API)
    - 文字生成 (一次性回應、串流回應與多輪對話)
    - 圖片生成
    - 語音生成
    - 文件解讀
    - 圖像解讀

