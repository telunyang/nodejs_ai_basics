/**
 * 文字生成 - 多輪對話
 * 
 * Gemini API 說明文件:
 * https://ai.google.dev/gemini-api/docs/text-generation?hl=zh-tw
 * 
 * config 參數說明:
 * https://ai.google.dev/api/generate-content?hl=zh-tw#v1beta.GenerationConfig
 * 
 */

// 匯入模組
import { GoogleGenAI } from "@google/genai";
import { get_gemini_api_key } from "./modules/myModule.mjs";

// 建立 GoogleGenAI 物件，並提供 API 金鑰
const ai = new GoogleGenAI({ apiKey:get_gemini_api_key() });

// 使用 GoogleGenAI 物件生成內容
async function main() {
    const chat = ai.chats.create({
        model: "gemini-2.0-flash-lite",
        history: [
            {
                role: "user",
                parts: [{ text: "你好" }],
            },
            {
                role: "model",
                parts: [{ text: "很高興見到你。你想知道什麼？" }],
            },
        ],
    });

    const response1 = await chat.sendMessage({
        message: "我家裡有兩隻狗。",
    });
    console.log("聊天回應 1:", response1.text);

    const response2 = await chat.sendMessage({
        message: "我家裡的狗有多少隻腳？",
    });
    console.log("聊天回應 2:", response2.text);
}

// 執行主函式
await main();