/**
 * 建立向量資料庫
 * 
 * 參考資料：
 * https://www.npmjs.com/package/@lancedb/lancedb
 * https://lancedb.github.io/lancedb/basic/
 * https://github.com/lancedb/lancedb
 * https://lancedb.com/docs/concepts/
 * 
 * Audio 來源：
 * https://mixkit.co/
 * 
 */

// 匯入模組
import { AutoProcessor, ClapAudioModelWithProjection, pipeline }  from '@huggingface/transformers';
import wavefile from 'wavefile';
import { readFileSync } from 'fs';
import * as lancedb from "@lancedb/lancedb";
import {
    Schema as ArrowSchema,
    Field,
    FixedSizeList,
    Int64,
    Utf8,
    Float32
} from 'apache-arrow';


// 建立文字特徵擷取管道
const pipe = await pipeline('feature-extraction', "Xenova/bge-m3", { dtype: 'auto' });

// 模型 ID
const model_id = "Xenova/clap-htsat-unfused";

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

// 平均池化函式
function mean_pooling(vector, dims) {
    /**
     * 將向量進行平均池化
     * @param {Float32Array} vector - 輸入向量
     * @param {Array} dims - 向量維度
     * @returns {Float32Array} - 平均池化後的向量
     */

    // 例如 vector: Float32Array(13312) [0.08195222169160843,    0.7124125361442566,  -0.9786397218704224, ... 13212 more items]
    // 例如 dims: [ 1, 13, 1024 ]
    const [batch, tokens, dim] = dims;

    // 將向量展開、攤平為一維陣列。目前 pooled 的內容是空的 Float32Array
    const pooled = new Float32Array(dim);

    // 檢查向量維度 (把原本攤平的 word embedding，重新組成 tokens[i] x dim 的矩陣)
    for (let i = 0; i < tokens; i++) {
        for (let j = 0; j < dim; j++) {
            pooled[j] = pooled[j] + vector[i * dim + j]; // 可以寫成 pooled[j] += vector[i * dim + j];
        }
    }

    // 計算平均值，將 pooled 中的每個特徵維度除以 tokens 的數量，這樣可以得到每個特徵維度的平均值。
    for (let j = 0; j < dim; j++) {
        pooled[j] = pooled[j] / tokens; // 也可以寫成: pooled[j] /= tokens;
    }

    return pooled;
}

// 準備寫入向量資料庫的資料
const arr_data = [
    {
        'id': 1,
        'text_desc': '熱烈的掌聲，可以感受到現場觀眾的歡呼與慶賀，氣氛熱烈歡騰，可能正在進行某種儀式或表演活動，人們正在為表演者或活動本身喝采。',
        'text_vector': null,
        'audio_path': './audios/3-1_0.wav',
        'audio_vector': null,
    },
    {
        'id': 2,
        'text_desc': '清晰的掌聲，密集而響亮，充滿了現場的活力和興奮。可能正在進行表演、演講或其他活動，觀眾正以熱烈的掌聲表達支持、讚賞或歡呼的情緒，渲染現場氣氛。',
        'text_vector': null,
        'audio_path': './audios/3-1_1.wav',
        'audio_vector': null,
    },
    {
        'id': 3,
        'text_desc': '清澈悅耳的吉他聲，伴隨著悠揚的旋律。吉他的音色溫柔而略帶憂鬱，可能是一段輕柔的演奏或背景音樂。整體給人一種放鬆、舒適，甚至有些許懷舊的氛圍。',
        'text_vector': null,
        'audio_path': './audios/3-1_2.wav',
        'audio_vector': null,
    },
    {
        'id': 4,
        'text_desc': '熱烈的歡呼聲，人們大聲尖叫、吶喊，氣氛相當興奮和激動。可以判斷現場可能正在進行某種激烈的活動、比賽，或者是一個非常精彩的表演，觀眾們的情緒高漲。',
        'text_vector': null,
        'audio_path': './audios/3-1_3.wav',
        'audio_vector': null,
    },
];

// 處理每個資料項目 (補上音訊特徵與文字特徵)
for (let i = 0; i < arr_data.length; i++) {
    const item = arr_data[i];

    // 讀取音訊檔案
    const buffer = readFileSync(item.audio_path);
    const audio_data = processAudio(buffer);

    // 將音訊數據轉換為模型輸入格式
    const audio_inputs = await audio_processor(audio_data);

    // 取得音訊特徵
    const audio_embeds = await audio_model(audio_inputs);

    // 儲存音訊特徵
    arr_data[i].audio_vector = Array.from(audio_embeds.audio_embeds.ort_tensor.cpuData);

    // 執行特徵擷取
    const text_feature = await pipe(item.text_desc);

    // 對文字特徵進行 mean pooling
    const pooled = mean_pooling(text_feature.ort_tensor.cpuData, text_feature.ort_tensor.dims);

    // 儲存文字特徵
    arr_data[i].text_vector = Array.from(pooled);
}


// 檢視變數內容
console.log(arr_data[0]);

// 建立向量資料庫
const db = await lancedb.connect("./db");

// 定義資料表結構
const schema = new ArrowSchema([
    new Field('id', new Int64(), false),
    new Field('text_desc', new Utf8(), false),
    new Field(
        'text_vector',
        new FixedSizeList(1024, new Field('item', new Float32(), false)),
        false  // 代表欄位為非空值 (non-null)
    ),
    new Field('audio_path', new Utf8(), true),
    new Field(
        'audio_vector',
        new FixedSizeList(512, new Field('item', new Float32(), false)),
        false // 代表欄位為非空值 (non-null)
    )
]);

// 建立資料表
const tbl = await db.createEmptyTable("knowledge", schema);

// 將資料寫入資料表
await tbl.add(arr_data);

console.log("資料表清單：", await db.tableNames());