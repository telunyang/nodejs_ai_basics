/**
 * 圖片特徵擷取與相似度計算
 * 
 * 參考模型: https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5
 */

// 匯入模組
import { pipeline } from '@huggingface/transformers';

// 建立圖片特徵擷取管道
const pipe = await pipeline('image-feature-extraction', 'nomic-ai/nomic-embed-vision-v1.5', {
    dtype: 'auto', // dtype: auto, fp32, fp16, q8, int8, uint8, q4, bnb4, q4f16
});

// 定義圖片路徑
const imagePath = [
    './images/1-1_0.jpg', // Image 0
    './images/1-1_1.jpg', // Image 1
    './images/1-1_2.jpg', // Image 2
    './images/1-1_3.jpg', // Image 3
];

// 計算 Cosine Similarity
function cos_sim(main_vector, other_vector) {
    /**
     * 計算兩個向量的餘弦相似度
     * @param {Array} main_vector - 主向量
     * @param {Array} other_vector - 其他向量
     * @returns {number} - 餘弦相似度
     */

    // 公式: 
    // cos(θ) = A·B / (||A|| * ||B||)
    // 其中 A·B 是兩個向量的點積，||A|| 和 ||B|| 是兩個向量的模長。
    // 這裡的點積是指兩個向量的內積，

    // 進階討論: 
    // 有沒有什麼額外的方法，讓兩個向量直接計算點積就能得到餘弦相似度，藉此簡化這個公式？

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

// 指定圖片的特徵 (zero-based index)
let index = 0; // 這裡可以修改為 0, 1, 2, 3 等，代表要用來比較的圖片索引
const main_vector = await pipe(imagePath[index]);
const second_vector = await pipe(imagePath[1]);

// 檢視變數內容
console.log(main_vector);
console.log(second_vector);

// 計算其他圖片與第 1 張的相似度
for (let i = 0; i < imagePath.length; i++) {
    // 跳過用來比較的圖片索引
    if (i === index) {
        continue;
    }

    // 獲取第 i 張圖片的特徵 (不包含第 1 張)
    const other_vector = await pipe(imagePath[i]);

    // 計算第 1 張與其它張圖片的餘弦相似度
    const similarity = cos_sim(main_vector.ort_tensor.cpuData, other_vector.ort_tensor.cpuData);

    // 輸出相似度結果
    console.log(`Image ${index} 與 Image ${i} 相似度: ${similarity.toFixed(4)}`);
}