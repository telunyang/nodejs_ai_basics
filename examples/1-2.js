/**
 * 物件偵測
 * 
 * 參考模型: https://huggingface.co/Xenova/detr-resnet-50
 */

// 匯入模組
import { pipeline } from '@huggingface/transformers';
import { writeFileSync } from 'fs';
import { registerFont, loadImage, createCanvas } from 'canvas';

// 註冊字型
// 注意：如果要顯示中文，請確保字型支援中文
registerFont('./fonts/NotoSansTC-Regular.ttf', { family: 'Noto Sans TC' });

// 建立物件偵測管道
const pipe = await pipeline('object-detection', 'Xenova/detr-resnet-50', { dtype: 'auto' });

// 定義圖片路徑
const imagePath = "./images/1-2_0.jpg"; // 單一圖片路徑

// 執行物件偵測
const results = await pipe(imagePath);

// 檢視變數內容
// console.log(results);

// 顯示結果
console.log("物件偵測結果:");
results.forEach((result, index) => {
    console.log(`物件 ${index + 1}:`);
    console.log(`  類別: ${result.label}`);
    console.log(`  信心度: ${result.score}`);
    console.log(`  位置: [${result.box.xmin}, ${result.box.ymin}, ${result.box.xmax}, ${result.box.ymax}]`);
});

// 將偵測結果繪製到圖片上
const drawResultsOnImage = async (imagePath, results) => {
    // 載入圖片並建立畫布
    const image = await loadImage(imagePath);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');

    // 繪製原始圖片
    ctx.drawImage(image, 0, 0);

    // 設定字型
    ctx.font = '20px "Arial"'; // 如果你有註冊中文字型，可以改成 '20px "Noto Sans TC"'
    ctx.textBaseline = 'top'; // 設定文字基線為頂部，這樣文字會從邊框上方開始繪製

    // 繪製每個偵測結果
    results.forEach(result => {
        const { xmin, ymin, xmax, ymax } = result.box;

        // 繪製紅色邊框
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

        // 標註文字（白底紅字提升可讀性）
        const label = `${result.label} (${(result.score * 100).toFixed(4)}%)`;
        const textWidth = ctx.measureText(label).width;

        // 背景白框
        ctx.fillStyle = 'white';
        ctx.fillRect(xmin, ymin - 22, textWidth + 6, 22);

        // 文字
        ctx.fillStyle = 'red';
        ctx.fillText(label, xmin + 3, ymin - 20);
    });

    // 儲存圖片
    const outputPath = './images/1-2_0_detected.jpg';
    const buffer = canvas.toBuffer('image/jpeg'); // 希望格式是 png 就改成 'image/png'
    writeFileSync(outputPath, buffer);
    console.log(`結果圖片已儲存至: ${outputPath}`);
};

// 繪製結果到圖片上
await drawResultsOnImage(imagePath, results);