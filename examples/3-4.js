/**
 * 文字轉語音
 * 
 * 參考模型: https://huggingface.co/Xenova/speecht5_tts
 * 參考聲碼器: https://huggingface.co/Xenova/speecht5_hifigan
 */

// ======== 簡易版本 ========
import { pipeline } from '@huggingface/transformers';
import wavefile from 'wavefile';
import { writeFileSync } from 'fs';

// 定義模型 ID
const model_id = 'Xenova/speecht5_tts';

// 初始化文字轉語音管道
const pipe = await pipeline('text-to-speech', model_id, { dtype: 'fp32' });

// 取得發聲者嵌入 (這是預先訓練好的發聲者嵌入)
const speaker_embeddings = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin';

// 建立輸入文本
let input_text = `Hello, my name is Melody, it's a pleasure to meet you! 
Today, I will be your guide through the world of speech synthesis.`;

// 生成語音
const results = await pipe(input_text, { speaker_embeddings });

// 檢視變數內容
// console.log(results);

// 將生成的語音轉換為 WAV 格式
const wav = new wavefile.WaveFile();
wav.fromScratch(1, results.sampling_rate, '32f', results.audio);
writeFileSync('./audios/3-4_output.wav', wav.toBuffer());


/*

// ======== 進階版本 ========

// 匯入模組
import { 
    AutoTokenizer, AutoProcessor, 
    SpeechT5ForTextToSpeech, SpeechT5HifiGan, 
    Tensor 
} from '@huggingface/transformers';
import wavefile from 'wavefile';
import { writeFileSync } from 'fs';

// 定義模型 ID
const model_id = 'Xenova/speecht5_tts';
const model_vocoder_id = 'Xenova/speecht5_hifigan';

// 讀取模型和處理器
// NOTE: 這裡使用的是 unquantized 版本，因為它們在精度上更好
// 如果需要更快的推理速度，可以考慮使用 quantized 版本
const tokenizer = await AutoTokenizer.from_pretrained(model_id);
const processor = await AutoProcessor.from_pretrained(model_id);

// 讀取模型和聲碼器
const model = await SpeechT5ForTextToSpeech.from_pretrained(model_id, { 
    quantized: false, 
    dtype: 'auto' 
});
const vocoder = await SpeechT5HifiGan.from_pretrained(model_vocoder_id, { 
    quantized: false, 
    dtype: 'auto' 
});

// 讀取發聲者嵌入 (因為模型需要發聲者嵌入來生成語音)
// 這裡使用的是一個預先訓練好的發聲者嵌入
// 如果需要使用自定義的發聲者嵌入，可以參考模型的文檔來生成自己的發聲者嵌入
const speaker_embeddings_data = new Float32Array(
    await (
        await fetch(
            'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin'
        )
    ).arrayBuffer()
);

// 將發聲者嵌入轉換為 Tensor
const speaker_embeddings = new Tensor(
    'float32',
    speaker_embeddings_data,
    [1, speaker_embeddings_data.length]
)

// 建立輸入文本
let input_text = `Hello, my name is Melody, it's a pleasure to meet you! 
Today, I will be your guide through the world of speech synthesis.`;

// 將輸入文本轉換為模型需要的格式，例如進行分詞和編碼
const inputs = await tokenizer(input_text);

// 生成語音 (waveform 變數名稱是固定的)
const { waveform } = await model.generate_speech(
    inputs['input_ids'],
    speaker_embeddings,
    { vocoder }
);

// 檢查變數內容
// console.log(waveform)

// 將生成的語音轉換為 WAV 格式
const wav = new wavefile.WaveFile();
wav.fromScratch(
    1, 
    processor.feature_extractor.config.sampling_rate, 
    '32f', 
    waveform.data
);
writeFileSync('./audios/3-4_output.wav', wav.toBuffer());

*/