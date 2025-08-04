/**
 * 將 mp3 轉換成 wav 檔案
 * 
 * 安裝指令 (解壓縮 zip 檔案):
 * npm i adm-zip --save
 * 
 * 語音測試網址: 
 * https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=zh-TW&q=你的自訂文字
 * 
 * 下載 ffmpeg:
 * https://ffmpeg.org/download.html
 * https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
 * 
 * 測試 mp3 轉 wav 的指令 (在 Windows 環境裡，路徑中的反斜線需要轉義成兩個反斜線):
 * ffmpeg\\bin\\ffmpeg.exe -i ./output_tts.mp3 -acodec pcm_s16le -ar 16000 -ac 1 ./output_tts.wav -y
 */

// 匯入
import { writeFileSync, existsSync, renameSync } from 'fs';
import fetch from 'node-fetch';
import AdmZip from 'adm-zip';
import { exec } from 'child_process';

// 下載 mp3 的網址
let text = "如果說再見...是妳唯一的消息...我彷彿可以預見我自己...越往遠處飛去...妳越在我心裡...而我卻是妳不要的回憶";
const url_tts = `https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=zh-TW&q=${text}`;

// 下載 mp3 檔案
const res_tts = await fetch(url_tts);

// 判斷如果 mp3 檔案不在，就進行下載
const mp3FilePath = './output_tts.mp3';
if (!existsSync(mp3FilePath)) {
    console.log("正在下載 mp3 檔案...");

    // 儲存成 mp3 檔案
    writeFileSync(mp3FilePath, await res_tts.buffer());

    console.log("mp3 檔案下載完成。");
    console.log(`mp3 儲存路徑: ${mp3FilePath}`);
} 

// 下載 ffmpeg 的預設檔案名稱
const ffmpegFilePath = 'ffmpeg-release-essentials.zip';

// 新的 ffmpeg 資料夾名稱
const newFfmpegDir = './ffmpeg';

// 判斷如果 ffmpeg 檔案不在，就進行下載
if (!existsSync(ffmpegFilePath)) {    
    console.log("正在下載 ffmpeg-release-essentials.zip ...");

    // 下載 ffmpeg
    const ffmpeg_url = `https://www.gyan.dev/ffmpeg/builds/${ffmpegFilePath}`;
    const res_ffmpeg = await fetch(ffmpeg_url);
    writeFileSync(ffmpegFilePath, await res_ffmpeg.buffer());

    // 解壓縮 ffmpeg
    const zip = new AdmZip(ffmpegFilePath);
    zip.extractAllTo('.', true);

    // 將 ffmpeg-xxxx-essentials 資料夾名稱改成 ffmpeg
    console.log(`正在重新命名 ${ffmpeg_url.split('/').pop().replace('.zip', '')} 資料夾，變成 ${newFfmpegDir} ...`);
    
    const ffmpegDir = zip.getEntries().find(entry => entry.entryName.startsWith('ffmpeg-')).entryName;
    renameSync(ffmpegDir, newFfmpegDir);
}

// 定義輸出 wav 檔案的路徑
const wavFilePath = './output_tts.wav';

// 如果 ffmpeg 資料夾存在，則執行 mp3 轉 wav 的指令
if (existsSync(newFfmpegDir)) {
    // 透過子程序執行 ffmpeg 指令將 mp3 轉換成 wav
    
    exec(`ffmpeg\\bin\\ffmpeg.exe -i ${mp3FilePath} -acodec pcm_s16le -ar 16000 -ac 1 ${wavFilePath} -y`);
    console.log("mp3 轉換成 wav 完成。");
}