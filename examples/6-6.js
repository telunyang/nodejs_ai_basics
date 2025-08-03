/**
 * 圖像解讀
 * 
 * Gemini API 說明文件:
 * https://ai.google.dev/gemini-api/docs/image-understanding?hl=zh-tw
 * 
 */

// 匯入模組
import { GoogleGenAI } from "@google/genai";
import { get_gemini_api_key } from "./modules/myModule.mjs";
import { readFileSync } from 'fs';

// 建立 GoogleGenAI 物件，並提供 API 金鑰
const ai = new GoogleGenAI({ apiKey: get_gemini_api_key() });

// 讀取圖片檔案並轉換為 base64 編碼
const base64ImageFile = readFileSync("./images/1-1_2.jpg", {encoding: "base64"});

// 準備內容，包含圖片的 base64 編碼和描述文字
const contents = [
    {
        inlineData: {
            mimeType: "image/jpeg",
            data: base64ImageFile,
        },
    },
    { text: "描述這張圖片" },
];

// 呼叫 Gemini API 生成內容
const response = await ai.models.generateContent({
    model: "gemini-2.0-flash-lite",
    contents: contents,
});

// 輸出回應內容
console.log(response.text);