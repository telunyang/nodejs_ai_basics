/**
 * 文字特徵擷取與相似度計算
 * 
 * 參考模型: https://huggingface.co/Xenova/bge-m3
 */

// 匯入模組
import { pipeline, AutoTokenizer } from '@huggingface/transformers';

// 定義模型ID
const model_id = 'Xenova/bge-m3';

// 初始化斷詞器
const tokenizer = await AutoTokenizer.from_pretrained(model_id);

// 建立特徵擷取管道
const pipe = await pipeline('feature-extraction', model_id, { dtype: 'auto' });

// 定義要處理的文字
const text1 = '感謝你的幫忙，這對我來說非常重要。';
const text2 = '謝謝你的協助，這對我來說意義重大。';

// 執行特徵擷取
const features1 = await pipe(text1);
const features2 = await pipe(text2);

// 檢視變數內容
console.log(features1);
console.log(features2);

// 檢視斷詞的列表
console.log("第 1 句的斷詞:");
console.log(tokenizer.tokenize(text1, { add_special_tokens: true }));
console.log("第 2 句的斷詞:");
console.log(tokenizer.tokenize(text2, { add_special_tokens: true }));

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

    /** 
     * 這裡的 vector[i * dim + j] 是指第 i 個 token 在第 j 個特徵維度的值。
     * 這樣的計算方式可以將每個 token 在每個特徵維度的值累加到 pooled 中。
     * 最終 pooled[j] 將包含所有 tokens 在第 j 個特徵維度的總和。
     * 這樣的累加方式可以確保每個特徵維度的值都被正確計算。
     * 例如 pooled[0] = vector[0 * dim + 0] + vector[1 * dim + 0] + ... + vector[(tokens - 1) * dim + 0]
     * 例如 pooled[1] = vector[0 * dim + 1] + vector[1 * dim + 1] + ... + vector[(tokens - 1) * dim + 1]
     * 這樣 pooled[0] 就是所有 tokens 在第 0 個特徵維度的總和。
     * 而 pooled[1] 就是所有 tokens 在第 1 個特徵維度的總和。
     * 以此類推，直到計算出 pooled[dim - 1]。
     */

    // 計算平均值，將 pooled 中的每個特徵維度除以 tokens 的數量，這樣可以得到每個特徵維度的平均值。
    for (let j = 0; j < dim; j++) {
        pooled[j] = pooled[j] / tokens; // 也可以寫成: pooled[j] /= tokens;
    }

    // 最終 pooled 將包含每個特徵維度的平均值，這樣就完成了平均池化的操作。
    // 例如 pooled: Float32Array(1024) [0.08195222169160843, 0.7124125361442566, -0.9786397218704224, ... 1021 more items]
    // 這樣 pooled 就是每個特徵維度的平均值，
    return pooled;
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

// 進行 mean pooling
const pooled1 = mean_pooling(features1.ort_tensor.cpuData, features1.ort_tensor.dims);
const pooled2 = mean_pooling(features2.ort_tensor.cpuData, features2.ort_tensor.dims);

// 計算相似度
const similarity = cos_sim(pooled1, pooled2);

// 輸出結果
console.log(`第 1 句和第 2 句的餘弦相似度為: ${similarity}`);