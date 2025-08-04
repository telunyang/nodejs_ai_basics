/**
 * Open Chinese Convert 開放中文轉換
 * 
 * 安裝指令:
 * npm i opencc-js --save
 * 
 * 網址:
 * https://www.npmjs.com/package/opencc-js
 * https://github.com/nk2028/opencc-js
 * 
 */

import * as OpenCC from 'opencc-js';

const converter = OpenCC.Converter({ from: 'cn', to: 'twp' });
let text = converter('人生也许会背叛你，兄弟会欺瞒你，只有数学不会 -- 数学不会就是不会。')
console.log(text);