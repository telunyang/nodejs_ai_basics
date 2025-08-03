/**
 * 文字分類
 * 
 * 參考模型: https://huggingface.co/Xenova/bert-base-multilingual-uncased-sentiment
 */

// 匯入模組
import { pipeline } from '@huggingface/transformers';

// 定義模型 ID
const model_id = 'Xenova/bert-base-multilingual-uncased-sentiment';

// 初始化分類管道
const pipe = await pipeline('sentiment-analysis', model_id, { dtype: 'auto' });

// 定義要分類的文字
const arr_text = [
    'That is a great movie!',
    'I am not happy with the service.',
    'The weather is terrible today.',
    'I love the new design of the product.',
    '今天的天氣真好！',
    '這部電影讓我感到很失望。'
];

// 執行分類
const results = await pipe(arr_text);

// 檢視變數內容
// console.log(results);

// 輸出結果
for (let i = 0; i < arr_text.length; i++) {
    console.log(`文字: ${arr_text[i]}`);
    console.log(`分類: ${results[i].label}`);
    console.log(`分數: ${results[i].score.toFixed(4)}`);
    console.log('-------------------------');
}