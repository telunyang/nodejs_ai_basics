/**
 * 圖片分類
 * 
 * 參考模型: https://huggingface.co/Xenova/vit-base-patch16-224
 */

// 匯入模組
import { pipeline } from '@huggingface/transformers';

// 建立圖片分類管道
const pipe = await pipeline('image-classification', 'Xenova/vit-base-patch16-224', { dtype: 'auto' });

// 定義圖片路徑
const imagePath = './images/1-3_0.jpg'; // 單一圖片路徑

// 執行圖片分類
const results = await pipe(imagePath);

// 檢視變數內容
// console.log(results);

// 顯示分類結果
results.forEach((item) => {
    console.log(`Label: ${item.label}\t\tScore: ${item.score.toFixed(4)}`);
});