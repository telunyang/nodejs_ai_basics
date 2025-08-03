/**
 * 圖片內容檢視
 * 
 * 參考模型: https://huggingface.co/onnx-community/Florence-2-base-ft
 */

// 匯入模組
import {
    Florence2ForConditionalGeneration,
    AutoProcessor,
    load_image,
    RawImage,
} from '@huggingface/transformers';
import path from 'path';

// 讀取模型與處理器
const model_id = 'onnx-community/Florence-2-base-ft';
const model = await Florence2ForConditionalGeneration.from_pretrained(model_id, { dtype: 'auto' });
const processor = await AutoProcessor.from_pretrained(model_id);

// 讀取網路上的圖片
// const url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg';
// const image = await load_image(url);

// 或者從本地檔案系統讀取圖片
const image_path = './images/1-2_0.jpg'; // 替換為你的圖片路徑
const absPath = path.resolve(image_path);
const image = await RawImage.read(absPath);

// 定義任務描述 (請參考以下列表中的一個選項)
/**
 * 參考連結:
 * https://huggingface.co/microsoft/Florence-2-base-ft
 * 
 * <OD>
 * <CAPTION>
 * <DETAILED_CAPTION>
 * <MORE_DETAILED_CAPTION>
 * <CAPTION_TO_PHRASE_GROUNDING> (要在 prompts 變數後面加一些描述)
 * <DENSE_REGION_CAPTION>
 * <REGION_PROPOSAL>
 * <OCR>
 * <OCR_WITH_REGION>
 */
// const task = '<CAPTION_TO_PHRASE_GROUNDING>';
const task = '<CAPTION>';
const prompts = processor.construct_prompts(task);

// 前處理圖片與提示
// 備註: 如果使用 <CAPTION_TO_PHRASE_GROUNDING> 任務，則需要在 prompts 之後加上描述，例如：
// const inputs = await processor(image, prompts + 'A remote on the left side of the image.');
const inputs = await processor(image, prompts);

// 生成文字 (尚未解碼)
const generated_ids = await model.generate({
    ...inputs,
    max_new_tokens: 100,
});

// 將生成的 ID 轉換為文字 (解碼)
const generated_text = processor.batch_decode(generated_ids, { skip_special_tokens: false })[0];

// 處後理生成的文字
const results = processor.post_process_generation(generated_text, task, image.size);

// 檢視變數內容
console.log(results);