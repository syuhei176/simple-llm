/**
 * 収集した文章をトレーニングデータに変換するスクリプト
 *
 * 使用方法:
 * npx ts-node scripts/prepare-training-data.ts <input-file> [output-file] [options]
 *
 * 例:
 * npx ts-node scripts/prepare-training-data.ts data/corpus.txt data/training-data.json --window 5 --stride 1
 */

import * as fs from 'fs';
import * as path from 'path';
import { convertTextToTrainingData } from '../src/training-data';

interface TrainingDataOptions {
  windowSize: number;
  stride: number;
  maxSamples?: number;
  minWordLength?: number;
}

function preprocessText(text: string, minWordLength: number = 2): string {
  // テキストのクリーニング
  return text
    .replace(/\r\n/g, '\n') // 改行を統一
    .replace(/\n+/g, ' ') // 複数の改行を空白に
    .replace(/\s+/g, ' ') // 複数の空白を1つに
    .replace(/[^\w\s.,!?;:'-]/g, '') // 特殊文字を除去
    .split(/\s+/)
    .filter(word => word.length >= minWordLength) // 短すぎる単語を除外
    .join(' ')
    .trim();
}

function convertToTrainingData(
  inputFile: string,
  outputFile: string,
  options: TrainingDataOptions
): void {
  console.log('=== Training Data Preparation ===\n');
  console.log(`Input file: ${inputFile}`);
  console.log(`Output file: ${outputFile}`);
  console.log(`Window size: ${options.windowSize}`);
  console.log(`Stride: ${options.stride}`);
  if (options.maxSamples) {
    console.log(`Max samples: ${options.maxSamples}`);
  }
  console.log('');

  // テキストファイルを読み込み
  if (!fs.existsSync(inputFile)) {
    console.error(`Error: Input file not found: ${inputFile}`);
    process.exit(1);
  }

  const rawText = fs.readFileSync(inputFile, 'utf-8');
  console.log(`Loaded ${rawText.length} characters from input file`);

  // テキストの前処理
  console.log('\nPreprocessing text...');
  const cleanText = preprocessText(rawText, options.minWordLength);
  console.log(`After preprocessing: ${cleanText.length} characters`);

  const wordCount = cleanText.split(/\s+/).length;
  console.log(`Word count: ${wordCount} words`);

  // スライディングウィンドウでトレーニングデータに変換
  console.log('\nConverting to training data...');
  let trainingData = convertTextToTrainingData(
    cleanText,
    options.windowSize,
    options.stride
  );

  // サンプル数を制限
  if (options.maxSamples && trainingData.length > options.maxSamples) {
    console.log(`Limiting to ${options.maxSamples} samples`);
    trainingData = trainingData.slice(0, options.maxSamples);
  }

  // 出力ディレクトリを作成
  const outputDir = path.dirname(outputFile);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // JSONファイルとして保存
  fs.writeFileSync(
    outputFile,
    JSON.stringify(trainingData, null, 2),
    'utf-8'
  );

  console.log(`\n✓ Successfully generated ${trainingData.length} training samples`);
  console.log(`✓ Saved to: ${outputFile}`);

  // サンプルを表示
  console.log('\nFirst 5 samples:');
  trainingData.slice(0, 5).forEach((sample, i) => {
    console.log(`${i + 1}. input: "${sample.input.substring(0, 50)}..."`);
    console.log(`   target: "${sample.target.substring(0, 50)}..."`);
  });

  // 統計情報
  const avgInputLength = trainingData.reduce((sum, s) => sum + s.input.length, 0) / trainingData.length;
  const avgTargetLength = trainingData.reduce((sum, s) => sum + s.target.length, 0) / trainingData.length;

  console.log('\nStatistics:');
  console.log(`  Total samples: ${trainingData.length}`);
  console.log(`  Average input length: ${avgInputLength.toFixed(1)} chars`);
  console.log(`  Average target length: ${avgTargetLength.toFixed(1)} chars`);
  console.log(`  File size: ${(fs.statSync(outputFile).size / 1024).toFixed(2)} KB`);
}

function parseArguments(): {
  inputFile: string;
  outputFile: string;
  options: TrainingDataOptions;
} {
  const args = process.argv.slice(2);

  if (args.length < 1) {
    console.log('Usage: npx ts-node scripts/prepare-training-data.ts <input-file> [output-file] [options]');
    console.log('\nOptions:');
    console.log('  --window <n>      Window size (default: 5)');
    console.log('  --stride <n>      Stride (default: 1)');
    console.log('  --max <n>         Maximum samples to generate');
    console.log('  --min-word <n>    Minimum word length (default: 2)');
    console.log('\nExamples:');
    console.log('  npx ts-node scripts/prepare-training-data.ts data/corpus.txt');
    console.log('  npx ts-node scripts/prepare-training-data.ts data/corpus.txt data/training.json --window 7 --stride 2');
    console.log('  npx ts-node scripts/prepare-training-data.ts data/corpus.txt data/training.json --max 1000');
    process.exit(1);
  }

  const inputFile = args[0];
  let outputFile = args[1] && !args[1].startsWith('--')
    ? args[1]
    : 'data/training-data.json';

  const options: TrainingDataOptions = {
    windowSize: 5,
    stride: 1,
    minWordLength: 2,
  };

  // オプションをパース
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--window' && args[i + 1]) {
      options.windowSize = parseInt(args[i + 1], 10);
    } else if (args[i] === '--stride' && args[i + 1]) {
      options.stride = parseInt(args[i + 1], 10);
    } else if (args[i] === '--max' && args[i + 1]) {
      options.maxSamples = parseInt(args[i + 1], 10);
    } else if (args[i] === '--min-word' && args[i + 1]) {
      options.minWordLength = parseInt(args[i + 1], 10);
    }
  }

  return { inputFile, outputFile, options };
}

// スクリプトとして実行された場合
if (require.main === module) {
  try {
    const { inputFile, outputFile, options } = parseArguments();
    convertToTrainingData(inputFile, outputFile, options);
  } catch (error) {
    console.error('Fatal error:', error);
    process.exit(1);
  }
}

export { convertToTrainingData, preprocessText };
