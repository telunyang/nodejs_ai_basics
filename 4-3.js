/**
 * 向量檢索 (使用音訊檔案)
 * 
 * 參考資料：
 * https://www.npmjs.com/package/@lancedb/lancedb
 * https://lancedb.com/docs/concepts/search/vector-search/
 * 
 */

// 匯入模組
import { AutoProcessor, ClapAudioModelWithProjection }  from '@huggingface/transformers';
import wavefile from 'wavefile';
import { readFileSync } from 'fs';
import * as lancedb from "@lancedb/lancedb";

// 初始化音訊特徵擷取模型
const audio_processor = await AutoProcessor.from_pretrained("Xenova/clap-htsat-unfused");
const audio_model = await ClapAudioModelWithProjection.from_pretrained("Xenova/clap-htsat-unfused", {
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

// 開啟向量資料庫
const db = await lancedb.connect("./db");

// 取得資料表
const tbl = await db.openTable("knowledge");

// 檢視資料筆數
const total = await tbl.countRows();
console.log(`資料總筆數：${total}`);

// 將整張表查出來 (請對大表加上 .limit() 以避免記憶體爆掉)
const rows = await tbl.query().limit(total).toArray();
console.table(rows);

// 取得範例音訊檔案
const audio_path = "./audios/4-3_0.wav";

// 使用音訊檔案進行向量檢索
let buffer = readFileSync(audio_path);
let audio = await processAudio(buffer);

// 將音訊數據轉換為模型輸入格式
const audio_inputs = await audio_processor(audio);

// 取得音訊特徵
const audio_embeds = await audio_model(audio_inputs);

// 執行向量檢索
const results = await tbl
    .search(audio_embeds.audio_embeds.ort_tensor.cpuData)
    .distanceType('cosine') // 預設是 'l2'，設定為 'cosine' 的話，要改成「1 - 距離」
    .select(['id', 'text_desc', 'audio_path', '_distance'])
    .limit(3)
    .toArray();

// 顯示檢索結果
console.log(`檢索到 ${results.length} 筆資料：`);
console.table(results);

// 顯示檢索結果
for (const result of results) {
    console.log(`文件編號：${result.id}`);
    console.log(`文字描述：${result.text_desc}`);
    console.log(`音訊路徑：${result.audio_path}`);
    console.log(`相似度：${1 - result._distance}`);
}
