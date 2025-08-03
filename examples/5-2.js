/**
 * 語音內容檢視
 * 
 * 參考模型: https://huggingface.co/onnx-community/ultravox-v0_5-llama-3_2-1b-ONNX
 */

// 匯入模組
import { UltravoxProcessor, UltravoxModel } from "@huggingface/transformers";
import wavefile from 'wavefile';

// 模型 ID
const model_id = "onnx-community/ultravox-v0_5-llama-3_2-1b-ONNX";

// 讀取模型與處理器
const processor = await UltravoxProcessor.from_pretrained(model_id);
const model = await UltravoxModel.from_pretrained(model_id, {
    dtype: {
        embed_tokens: "q8", // "fp32", "fp16", "q8"
        audio_encoder: "q4", // "fp32", "fp16", "q8", "q4", "q4f16"
        decoder_model_merged: "q4", // "q8", "q4", "q4f16"
    },
});

// 建立函式
function processAudio(buffer) {
    // 將 Buffer 轉換為 WaveFile
    let wav = new wavefile.WaveFile(buffer);
    wav.toBitDepth('32f'); // 將輸入的音訊轉換為 32 位元浮點數格式 (Float32)
    wav.toSampleRate(16000); // 轉換成 16kHz 的採樣率
    let audioData = wav.getSamples();

    // 如果音訊是多通道的，則將其轉換為單通道
    if (Array.isArray(audioData)) {
        if (audioData.length > 1) {
            const SCALING_FACTOR = Math.sqrt(2);

            // 合併頻道來節省記憶體
            for (let i = 0; i < audioData[0].length; ++i) {
                audioData[0][i] = SCALING_FACTOR * (audioData[0][i] + audioData[1][i]) / 2;
            }
        }

        // 選擇第一個頻道的音訊數據
        audioData = audioData[0];
    }

    return audioData;
}

// 讀取網路上的音訊檔案
const url = "http://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/mlk.wav";
let buffer = Buffer.from(await fetch(url).then(x => x.arrayBuffer()))

// 或者從本地檔案系統讀取音訊檔案
// const url = "./audios/5-2_0.wav";
// let buffer = readFileSync(url);

// 處理音訊
const audio = processAudio(buffer);

// 定義訊息
const messages = [
    {role: "system", content: "You are a helpful assistant."},
    {role: "user", content: "Transcribe this audio:<|audio|>"},
];

// 將訊息轉換為文字
const text = processor.tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
});

// 將音訊與文字結合
const inputs = await processor(text, audio);

// 生成文字 (尚未解碼)
const generated_ids = await model.generate({
    ...inputs,
    max_new_tokens: 128,
});

// 將生成的 ID 轉換為文字 (解碼)
const generated_texts = processor.batch_decode(
    generated_ids.slice(
        null, 
        [inputs.input_ids.dims.at(-1), null]
    ),
    { skip_special_tokens: true },
);

// 檢視變數內容
console.log(generated_texts);