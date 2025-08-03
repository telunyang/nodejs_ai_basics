/**
 * 文字生成 - 一次性回應與串流回應
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
    /**
     * 一般回應
     */

    // 使用模型生成內容
    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash-lite",
        contents: "用 100 個字來描述 AI 的運作方式。",
        config: {
            systemInstruction: "You are a helpful assistant.", // 系統提示，設定模型的角色或任務
            thinkingConfig: {
                thinkingBudget: 0, // 0 表示不讓模型使用思考模式
            },
            maxOutputTokens: 150, // 最大輸出字數
            stopSequences: ["\n", "。"], // 停止生成的序列
            temperature: 0.7, // 控制生成文本的隨機性，越高越隨機
            topK: 50, // 前 K 個最可能的詞
            topP: 0.9, // 控制生成文本的多樣性 (累進機率)
            seed: 42, // 隨機種子，用於生成可重複的結果
        }
    });

    // 輸出生成的內容
    console.log(response.text);



    /**
     * 串流回應 (沒用到的話，可以先將以下程式碼註解掉)
     */

    // 使用模型生成內容的串流回應
    // 這樣可以逐步接收生成的文本，而不是一次性獲得完整文字
    // const response = await ai.models.generateContentStream({
    //     model: "gemini-2.0-flash-lite",
    //     contents: "用一段話來描述 AI 的運作方式。",
    // });

    // 使用 for-await-of 迴圈來處理串流回應
    // 每次迭代都會獲得一個新的文字片段
    // for await (const chunk of response) {
    //     console.log(chunk.text);
    // }
}

// 執行主函式
await main();