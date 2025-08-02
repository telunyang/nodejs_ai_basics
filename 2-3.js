/**
 * 情感分析
 * 
 * 參考模型: https://huggingface.co/Xenova/distilroberta-finetuned-financial-news-sentiment-analysis
 */

// 匯入模組
import { pipeline } from '@huggingface/transformers';

// 定義模型 ID
const model_id = 'Xenova/distilroberta-finetuned-financial-news-sentiment-analysis';

// 初始化分類管道
const pipe = await pipeline('sentiment-analysis', model_id, { dtype: 'auto' });

// 定義要分類的文字
const arr_text = [
    'The company reported a profit increase in the last quarter.',
    'Stock prices fell sharply after the announcement.',
    'Investors are optimistic about the new product launch.',
    'The economic outlook remains uncertain amid rising inflation.',
    'I have not decided whether to invest in this stock yet.',
    'The market reacted negatively to the CEO\'s resignation.',
    'My family and I are planning a vacation next month.',
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