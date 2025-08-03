/**
 * 語音生成
 * 
 * Gemini API 說明文件:
 * https://ai.google.dev/gemini-api/docs/speech-generation?hl=zh-tw
 * 
 * 如果出現「Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'wav' imported from D:\teach\nodejs_ai_basics\6-4.js」
 * npm install wav --legacy-peer-deps
 */

// 匯入模組
import { GoogleGenAI } from "@google/genai";
import { get_gemini_api_key } from "./modules/myModule.mjs";
import wav from 'wav';

// 建立 GoogleGenAI 物件，並提供 API 金鑰
const ai = new GoogleGenAI({ apiKey: get_gemini_api_key() });

// 使用 wav 模組來寫入 PCM 資料到 WAV 檔案
async function saveWaveFile(filename, pcmData, channels = 1, rate = 24000, sampleWidth = 2) {
   return new Promise((resolve, reject) => {
        // 建立 WAV 檔案寫入器
        const writer = new wav.FileWriter(filename, {
            channels,
            sampleRate: rate,
            bitDepth: sampleWidth * 8,
        });

        // 當寫入完成時，解析 Promise
        writer.on('finish', resolve);

        // 當發生錯誤時，拒絕 Promise
        writer.on('error', reject);

        // 將 PCM 資料寫入 WAV 檔案
        writer.write(pcmData);

        // 關閉寫入器
        writer.end();
   });
}

// 主函式，生成語音並儲存為 WAV 檔案
async function main() {
    // 呼叫 Gemini API 生成語音內容
    const response = await ai.models.generateContent({
        model: "gemini-2.5-flash-preview-tts",
        contents: [
            { parts: [
                { text: 'Say cheerfully: Have a wonderful day!' }
            ]}
        ],
        config: {
            responseModalities: ['AUDIO'],
            speechConfig: {
                voiceConfig: {
                    // Voice 選項:
                    // https://ai.google.dev/gemini-api/docs/speech-generation#voices
                    prebuiltVoiceConfig: { voiceName: 'Kore' },
                },
            },
        },
    });

    // 取得音訊資料並轉換為 Buffer
    // 備註:「?.」是 Optional Chaining，確保在物件不存在時不會拋出錯誤，
    // 用來安全地存取巢狀物件的屬性，即使中間某一層是 null 或 undefined 也不會拋錯，
    // 而是直接傳回 undefined。
    const data = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;  
    const audioBuffer = Buffer.from(data, 'base64');

    // 儲存音訊為 WAV 檔案
    const fileName = './audios/out.wav';
    await saveWaveFile(fileName, audioBuffer);
}

// 執行主函式
await main();