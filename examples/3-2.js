/**
 * 語音轉文字
 * 
 * 參考模型: https://huggingface.co/Xenova/whisper-medium
 * 語音測試網址: https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=zh-TW&q=你的自訂文字
 * 範例: 如果說再見...是妳唯一的消息...我彷彿可以預見我自己...越往遠處飛去...妳越在我心裡...而我卻是妳不要的回憶
 */

// 匯入模組
import { pipeline } from '@huggingface/transformers';
import wavefile from 'wavefile';
import { readFileSync } from 'fs';

// 定義模型 ID
const model_id = 'Xenova/whisper-medium';

// 初始化語音轉文字管道 (如果用 CPU 可能會很久，建議使用 GPU)
const pipe = await pipeline('automatic-speech-recognition', model_id, {
    dtype: 'auto', 
    device: 'cpu'
});

// 取得語音檔案 (如果是 mp3，需要先轉換成 wav 格式)
const url = "./audios/3-2_0.wav";

// 如果需要從網路下載語音檔案，可以使用以下程式碼
// let buffer = Buffer.from(await fetch(url).then(x => x.arrayBuffer()))

// 如果語音檔案已經存在於本地，可以直接讀取
let buffer = readFileSync(url);

// 將 Buffer 轉換為 WaveFile
let wav = new wavefile.WaveFile(buffer);
wav.toBitDepth('32f'); // 將輸入的音訊轉換為 32 位元浮點數格式 (Float32)
wav.toSampleRate(16000); // 轉換成 16kHz 的採樣率，給 Whisper 模型使用
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

// 計算處理時間 - 開始計時
const start = Date.now();

/**
 * language 列表:
 * ["english","chinese","german","spanish",
 *  "russian","korean","french","japanese",
 *  "portuguese","turkish","polish","catalan",
 *  "dutch","arabic","swedish","italian",
 *  "indonesian","hindi","finnish","vietnamese",
 *  "hebrew","ukrainian","greek","malay","czech",
 *  "romanian","danish","hungarian","tamil",
 *  "norwegian","thai","urdu","croatian",
 *  "bulgarian","lithuanian","latin","maori",
 *  "malayalam","welsh","slovak","telugu",
 *  "persian","latvian","bengali","serbian",
 *  "azerbaijani","slovenian","kannada","estonian",
 *  "macedonian","breton","basque","icelandic",
 *  "armenian","nepali","mongolian","bosnian",
 *  "kazakh","albanian","swahili","galician",
 *  "marathi","punjabi","sinhala","khmer",
 *  "shona","yoruba","somali","afrikaans",
 *  "occitan","georgian","belarusian","tajik",
 *  "sindhi","gujarati","amharic","yiddish",
 *  "lao","uzbek","faroese","haitian creole",
 *  "pashto","turkmen","nynorsk","maltese",
 *  "sanskrit","luxembourgish","myanmar","tibetan",
 *  "tagalog","malagasy","assamese","tatar",
 *  "hawaiian","lingala","hausa","bashkir","javanese",
 *  "sundanese"]
 */

// 取得語音轉文字的結果
let results = await pipe(audioData, {
    return_timestamps: true,    // 回傳時間戳記
    language: 'chinese'         // 輸出語系設定為中文
});

// 計算處理時間 - 結束計時
const end = Date.now();

// 計算總處理時間
const duration = end - start;

// 輸出處理時間
console.log(`處理時間: ${duration / 1000} 秒`);

// 檢視變數內容
console.log(results);

// 輸出語音轉文字的結果
console.log(`語音轉文字結果: ${results.text}`);

// 輸出 chunks 的內容
console.log('語音轉文字的 chunks:');
results.chunks.forEach((chunk, index) => {
    console.log('---------------------------------');
    console.log(`Chunk ${index + 1}:`);
    console.log(`[start: ${chunk.timestamp[0]}, end: ${chunk.timestamp[1]}] text: ${chunk.text}`);
});