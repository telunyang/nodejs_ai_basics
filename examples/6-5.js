/**
 * 文件解讀
 * 
 * Gemini API 說明文件:
 * https://ai.google.dev/gemini-api/docs/document-processing?hl=zh-tw#inline_data
 * 
 */

// 匯入模組
import { GoogleGenAI } from "@google/genai";
import { get_gemini_api_key } from "./modules/myModule.mjs";
import { readFileSync } from 'fs';

// 建立 GoogleGenAI 物件，並提供 API 金鑰
const ai = new GoogleGenAI({ apiKey: get_gemini_api_key() });

// 建立主函式
async function main() {
    // 讀取 PDF 檔案並轉換為 Base64 編碼
    const contents = [
        { text: "對這份文件進行摘要" },
        {
            inlineData: {
                mimeType: 'application/pdf',
                data: Buffer.from(readFileSync("files/6-5_0.pdf")).toString("base64")
            }
        }
    ];

    // 呼叫 Gemini API 生成內容
    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash-lite",
        contents: contents
    });

    // 輸出回應內容
    console.log(response.text);
}

// 執行主函式
await main();