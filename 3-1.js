/**
 * 音訊特徵擷取與相似度計算
 * 
 * 參考模型: https://huggingface.co/Xenova/clap-htsat-unfused
 * CLAP: Contrastive Language-Audio Pretraining
 */

// 匯入模組
import { AutoProcessor, ClapAudioModelWithProjection, read_audio }  from '@huggingface/transformers';
import wavefile from 'wavefile';
import { readFileSync } from 'fs';

// 定義模型 ID
const model_id = 'Xenova/clap-htsat-unfused';

// 初始化音訊特徵擷取模型
const audio_processor = await AutoProcessor.from_pretrained(model_id);
const audio_model = await ClapAudioModelWithProjection.from_pretrained(model_id, {
    dtype: 'auto', 
    device: 'cpu'
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

// 計算 Cosine Similarity
function cos_sim(main_vector, other_vector) {
    /**
     * 計算兩個向量的餘弦相似度
     * @param {Array} main_vector - 主向量
     * @param {Array} other_vector - 其他向量
     * @returns {number} - 餘弦相似度
     */

    // 檢查向量長度是否一致
    if (main_vector.length !== other_vector.length) {
        throw new Error('向量維度不一致，無法計算餘弦相似度');
    }

    // 計算點積和內積
    let dot_product = 0;
    let inner_product_A = 0;
    let inner_product_B = 0;

    // 向量計算
    for (let i = 0; i < main_vector.length; i++) {
        dot_product += main_vector[i] * other_vector[i];
        inner_product_A += main_vector[i] * main_vector[i];
        inner_product_B += other_vector[i] * other_vector[i];
    }

    // 計算餘弦相似度
    return dot_product / (Math.sqrt(inner_product_A) * Math.sqrt(inner_product_B));
}

// 定義音訊路徑
const audioPath = [
    './audios/3-1_0.wav', // Audio 0
    './audios/3-1_1.wav', // Audio 1
    './audios/3-1_2.wav',  // Audio 2
    './audios/3-1_3.wav',  // Audio 3
];

// 如果需要從網路下載語音檔案，可以使用以下程式碼
// let buffer = Buffer.from(await fetch(url).then(x => x.arrayBuffer()))

// 指定音訊的特徵 (zero-based index)
let index = 0; // 這裡可以修改為 0, 1, 2 等，代表要用來比較的音訊索引

// 如果語音檔案已經存在於本地，可以直接讀取
let buffer_main = readFileSync(audioPath[index]);

// 讀取音訊檔案
let audio_main = await processAudio(buffer_main);

// 將音訊數據轉換為模型輸入格式
const audio_inputs_main = await audio_processor(audio_main);

// 取得音訊特徵
const audio_embeds_main = await audio_model(audio_inputs_main);

// 檢視變數內容
// console.log(audio_embeds_main)

for (let i = 0; i < audioPath.length; i++) {
    // 跳過用來比較的音訊索引
    if (i === index) {
        continue;
    }

    // 讀取第 i 張音訊檔案
    let buffer_other = readFileSync(audioPath[i]);
    let audio_other = await processAudio(buffer_other);

    // 將音訊數據轉換為模型輸入格式
    const audio_inputs_other = await audio_processor(audio_other);

    // 取得音訊特徵
    const audio_embeds_other = await audio_model(audio_inputs_other);

    // 計算第 1 張與其它張音訊的餘弦相似度
    const similarity = cos_sim(
        audio_embeds_main.audio_embeds.ort_tensor.cpuData, 
        audio_embeds_other.audio_embeds.ort_tensor.cpuData
    );

    // 輸出相似度結果
    console.log(`Audio ${index} 與 Audio ${i} 的相似度: ${similarity}`);
}


