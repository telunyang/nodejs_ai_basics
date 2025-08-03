/**
 * 文字轉音樂
 * 
 * 參考模型: https://huggingface.co/Xenova/musicgen-small
 */

// 匯入模組
import { AutoTokenizer, MusicgenForConditionalGeneration, RawAudio } from '@huggingface/transformers';

// 定義模型 ID
const model_id = 'Xenova/musicgen-small';

// 初始化模型
const tokenizer = await AutoTokenizer.from_pretrained(model_id);
const model = await MusicgenForConditionalGeneration.from_pretrained(model_id, {
    dtype: {
        text_encoder: 'q8', // 文本編碼器使用 8 位元整數
        decoder_model_merged: 'q8', // 解碼器合併使用 8 位元整數
        encodec_decode: 'fp32' // 編碼器解碼器使用 32 位元浮點數
    },
    device: 'cpu' // 使用 CPU
});

// 準備輸入文字
// const prompt = 'a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130';
const prompt = 'lo-fi music with a soothing melody';
const inputs = await tokenizer(prompt);

// 計算處理時間 - 開始計時
const start = Date.now();

// 生成音樂
const audio_values = await model.generate({
    ...inputs,           //「...」是展開運算子，將 inputs 物件的屬性展開為函數參數
    do_sample: true,     // 使用隨機抽樣法
    temperature: 0.5,    // 溫度控制生成的隨機性
    max_new_tokens: 500, // 最大生成的 token 數量
    guidance_scale: 7.0, // 引導尺度，控制生成的多樣性，愈小愈多樣化，最小值為 1.0
});

// 計算處理時間 - 結束計時
const end = Date.now();

// 計算總處理時間
const duration = end - start;

// 輸出處理時間
console.log(`處理時間: ${duration / 1000} 秒`);

// 儲存成 wav 檔案
const audio = new RawAudio(audio_values.data, model.config.audio_encoder.sampling_rate);
audio.save('./audios/3-3_output.wav');