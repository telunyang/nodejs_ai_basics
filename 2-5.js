/**
 * 翻譯
 * 
 * 參考模型: https://huggingface.co/Xenova/nllb-200-distilled-600M
 * 語系代碼: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
 */

// 匯入模組
import { pipeline } from '@huggingface/transformers';

// 定義模型 ID
const model_id = 'Xenova/nllb-200-distilled-600M';

// 初始化翻譯管道
const pipe = await pipeline('translation', model_id, { dtype: 'auto' });

// 定義要翻譯的訊息
const results = await pipe('Hello, how is it going today?', {
    src_lang: 'eng_Latn', // 原始語言設定為英文
    tgt_lang: 'zho_Hant', // 目標語言設定為 Chinese (Traditional)
});

// 檢視變數內容
console.log(results);

// 輸出翻譯結果
console.log(results[0].translation_text);