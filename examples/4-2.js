/**
 * 向量檢索
 * 
 * 參考資料：
 * https://www.npmjs.com/package/@lancedb/lancedb
 * https://lancedb.com/docs/concepts/search/vector-search/
 * 
 */

// 匯入模組
import { pipeline }  from '@huggingface/transformers';
import * as lancedb from "@lancedb/lancedb";

// 建立文字特徵擷取管道
const pipe = await pipeline('feature-extraction', "Xenova/bge-m3", { dtype: 'auto' });

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


// 初始化搜尋文字
// let search_text = "清澈悅耳的吉他聲，伴隨著悠揚的旋律。吉他的音色溫柔而略帶憂鬱，可能是一段輕柔的演奏或背景音樂。整體給人一種放鬆、舒適，甚至有些許懷舊的氛圍。";
let search_text = "柔和清亮的吉他聲輕輕響起，旋律悠揚而帶有一點淡淡的哀愁，彷彿一段輕音樂，營造出放鬆又懷舊的氣息。";

// 使用文字進行向量檢索
const text_inputs = await pipe(search_text);
const text_vector = mean_pooling(text_inputs.ort_tensor.cpuData, text_inputs.ort_tensor.dims);

// 執行向量檢索
const results = await tbl
    .search(text_vector)
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
