/**
 * open-google-translator
 * 
 * 安裝指令:
 * npm i open-google-translator --save
 * 
 * 網址:
 * https://github.com/vidya-hub/open-google-translator
 * https://www.npmjs.com/package/open-google-translator
 * 
 * 語系可以參考:
 * https://zh.wikipedia.org/zh-tw/ISO_639-1
 * 
 */
import translator from "open-google-translator";

// 英文翻譯成中文
const results = await translator.TranslateLanguageData({
    listOfWordsToTranslate: ["How have you been?", "I am fine, thank you!"],
    fromLanguage: "en",
    toLanguage: "zh",
});
console.log(results);

// 中文翻譯成英文
// const results = await translator.TranslateLanguageData({
//     listOfWordsToTranslate: ["你最近怎麼樣?", "我很好，謝謝你!"],
//     fromLanguage: "zh",
//     toLanguage: "en",
// });
// console.log(results);