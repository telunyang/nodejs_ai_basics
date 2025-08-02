/**
 * 文字生成
 * 
 * 參考模型: https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct
 */

// 匯入模組
import { pipeline } from '@huggingface/transformers';

// 定義模型 ID
const model_id = 'onnx-community/Llama-3.2-1B-Instruct';

// 初始化生成管道
const pipe = await pipeline('text-generation', model_id, { dtype: 'auto' });

// 定義要生成的訊息
const messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
];

// 執行生成
const results = await pipe(messages, {
    max_new_tokens: 100,
    do_sample: true,
    temperature: 0.7,
    top_k: 50,
    top_p: 0.95
});

// 檢視變數內容
// console.log(results);
// console.log(results[0].generated_text);

// 輸出結果
console.log('生成的文字:');
const arr_text = results[0].generated_text;

// 取得最後一個生成的文字
console.log(arr_text[arr_text.length - 1].content);