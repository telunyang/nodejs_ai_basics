/**
 * 圖片生成
 * 
 * Gemini API 說明文件:
 * https://ai.google.dev/gemini-api/docs/image-generation?hl=zh-tw
 * 
 * config 參數說明:
 * https://ai.google.dev/api/generate-content?hl=zh-tw#v1beta.GenerationConfig
 * 
 */

// 匯入模組
import { GoogleGenAI, Modality  } from "@google/genai";
import { get_gemini_api_key } from "./modules/myModule.mjs";
import * as fs from "node:fs";

// 建立 GoogleGenAI 物件，並提供 API 金鑰
const ai = new GoogleGenAI({ apiKey: get_gemini_api_key() });

async function main() {
    // 定義要生成的內容。如果要取得較好的結果，請使用下列語言：
    // 英文、西班牙文 (墨西哥)、日文、中文 (中國大陸)、印地文 (印度)。
    const contents =
        "Hi, can you create a 3d rendered image of a pig " +
        "with wings and a top hat flying over a happy " +
        "futuristic scifi city with lots of greenery?";

    // 設定 responseModalities 包含 "Image"，讓模型可以生成圖片
    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash-preview-image-generation",
        contents: contents,
        config: {
            responseModalities: [Modality.TEXT, Modality.IMAGE],
        },
    });

    // 輸出生成的內容
    for (const part of response.candidates[0].content.parts) {
        // 基於部分類型，顯示文本或保存圖片
        if (part.text) {
            console.log(part.text);
        } else if (part.inlineData) {
            // 取得圖片數據
            const imageData = part.inlineData.data;

            // 將 base64 編碼的圖片數據轉換為 Buffer
            const buffer = Buffer.from(imageData, "base64");

            // 將圖片數據寫入檔案
            fs.writeFileSync("./images/gemini-native-image.png", buffer);

            // 輸出圖片保存成功的訊息
            console.log("Image saved as ./images/gemini-native-image.png");
        }
    }
}

// 執行主函式
await main();