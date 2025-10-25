"use strict";
/**
 * スライディングウィンドウ変換の使用例
 *
 * このファイルは、長い文章をトレーニングデータに変換する方法を示します。
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.example1 = example1;
exports.example2 = example2;
exports.example3 = example3;
exports.example4 = example4;
exports.example5 = example5;
const training_data_1 = require("../src/training-data");
const llm_1 = require("../src/llm");
// 例1: 単一の長い文章を変換
function example1() {
    console.log('=== 例1: 単一の文章のスライディングウィンドウ変換 ===\n');
    const longText = `
    The quick brown fox jumps over the lazy dog.
    This sentence contains every letter of the alphabet.
    Machine learning is a fascinating field of study.
    Neural networks can learn patterns from data.
  `;
    // ウィンドウサイズ5、ストライド1で変換
    const trainingData = (0, training_data_1.convertTextToTrainingData)(longText, 5, 1);
    console.log(`生成されたトレーニングデータ数: ${trainingData.length}\n`);
    console.log('最初の5サンプル:');
    trainingData.slice(0, 5).forEach((sample, i) => {
        console.log(`${i + 1}. input: "${sample.input}"`);
        console.log(`   target: "${sample.target}"\n`);
    });
}
// 例2: ウィンドウサイズとストライドを変えて比較
function example2() {
    console.log('\n=== 例2: ウィンドウサイズとストライドの比較 ===\n');
    const text = 'the cat sat on the mat and the dog sat on the rug';
    console.log('元の文章:', text);
    console.log(`単語数: ${text.split(/\s+/).length}\n`);
    // パターン1: ウィンドウ3、ストライド1（高密度）
    console.log('--- パターン1: ウィンドウ3、ストライド1 ---');
    const data1 = (0, training_data_1.convertTextToTrainingData)(text, 3, 1);
    console.log(`生成サンプル数: ${data1.length}`);
    data1.slice(0, 3).forEach((s, i) => {
        console.log(`${i + 1}. "${s.input}" → "${s.target}"`);
    });
    // パターン2: ウィンドウ5、ストライド2（中密度）
    console.log('\n--- パターン2: ウィンドウ5、ストライド2 ---');
    const data2 = (0, training_data_1.convertTextToTrainingData)(text, 5, 2);
    console.log(`生成サンプル数: ${data2.length}`);
    data2.forEach((s, i) => {
        console.log(`${i + 1}. "${s.input}" → "${s.target}"`);
    });
    // パターン3: ウィンドウ7、ストライド3（低密度）
    console.log('\n--- パターン3: ウィンドウ7、ストライド3 ---');
    const data3 = (0, training_data_1.convertTextToTrainingData)(text, 7, 3);
    console.log(`生成サンプル数: ${data3.length}`);
    data3.forEach((s, i) => {
        console.log(`${i + 1}. "${s.input}" → "${s.target}"`);
    });
}
// 例3: 複数の文章を一括変換
function example3() {
    console.log('\n=== 例3: 複数の文章を一括変換 ===\n');
    const texts = [
        'artificial intelligence is transforming the world',
        'deep learning models require large amounts of data',
        'natural language processing enables machines to understand human language',
    ];
    const trainingData = (0, training_data_1.convertMultipleTextsToTrainingData)(texts, 4, 2);
    console.log('\n生成されたトレーニングデータ:');
    trainingData.forEach((sample, i) => {
        console.log(`${i + 1}. "${sample.input}" → "${sample.target}"`);
    });
}
// 例4: 実際のモデル学習での使用
async function example4() {
    console.log('\n=== 例4: モデル学習での使用 ===\n');
    // 長い文章からトレーニングデータを生成
    const longText = `
    hello my friend how are you doing today.
    I am doing very well thank you for asking.
    what would you like to talk about today.
    I would like to learn about artificial intelligence.
    that is a fascinating topic to discuss.
  `;
    // スライディングウィンドウで変換（ウィンドウ5、ストライド1）
    const slidingWindowData = (0, training_data_1.convertTextToTrainingData)(longText, 5, 1);
    console.log(`生成されたトレーニングデータ数: ${slidingWindowData.length}`);
    // 語彙を作成
    const vocab = (0, training_data_1.createVocab)(slidingWindowData);
    console.log(`\n語彙サイズ: ${vocab.length}`);
    // モデルを作成
    const llm = new llm_1.SimpleLLM(vocab, 16, 2);
    console.log('\nモデルの学習を開始...');
    llm.train(slidingWindowData, 5);
    console.log('\n学習完了！モデルでテスト生成:');
    const testInput = 'hello my friend';
    const output = llm.predict(testInput, 5);
    console.log(`入力: "${testInput}"`);
    console.log(`出力: "${output}"`);
}
// 例5: 推奨設定のガイド
function example5() {
    console.log('\n=== 例5: 推奨設定ガイド ===\n');
    console.log('ウィンドウサイズの選び方:');
    console.log('- 小さい (3-5): 短い文脈、高速学習、多数のサンプル');
    console.log('- 中程度 (5-10): バランスの取れた文脈と学習速度');
    console.log('- 大きい (10+): 長い文脈、文章の流れを学習\n');
    console.log('ストライドの選び方:');
    console.log('- ストライド1: 最大のデータ効率、重複が多い');
    console.log('- ストライド = ウィンドウサイズ/2: バランスの取れた重複');
    console.log('- ストライド = ウィンドウサイズ: 重複なし、データ効率低い\n');
    console.log('実践的な推奨設定:');
    console.log('- 小規模データ (100-500単語): ウィンドウ3-5、ストライド1');
    console.log('- 中規模データ (500-2000単語): ウィンドウ5-7、ストライド2');
    console.log('- 大規模データ (2000+単語): ウィンドウ7-10、ストライド3');
}
// すべての例を実行
async function runAllExamples() {
    example1();
    example2();
    example3();
    await example4();
    example5();
}
// Node.js環境で直接実行された場合
if (require.main === module) {
    runAllExamples().catch(console.error);
}
