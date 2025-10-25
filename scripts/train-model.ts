/**
 * トレーニングデータからモデルを学習し保存するスクリプト
 *
 * 使用方法:
 * npx ts-node scripts/train-model.ts <training-data-file> [model-name] [options]
 *
 * 例:
 * npx ts-node scripts/train-model.ts data/training-data.json my-model --epochs 50
 */

import * as fs from 'fs';
import * as path from 'path';
import { SimpleLLM } from '../src/llm';
import { createVocab } from '../src/training-data';

interface TrainingOptions {
  epochs: number;
  embeddingDim: number;
  numLayers: number;
  modelName: string;
}

function loadTrainingData(filePath: string): { input: string; target: string }[] {
  if (!fs.existsSync(filePath)) {
    console.error(`Error: Training data file not found: ${filePath}`);
    process.exit(1);
  }

  const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

  if (!Array.isArray(data) || data.length === 0) {
    console.error('Error: Invalid training data format');
    process.exit(1);
  }

  return data;
}

function trainModel(
  trainingData: { input: string; target: string }[],
  options: TrainingOptions
): SimpleLLM {
  console.log('=== Model Training ===\n');
  console.log(`Training data: ${trainingData.length} samples`);
  console.log(`Epochs: ${options.epochs}`);
  console.log(`Embedding dimension: ${options.embeddingDim}`);
  console.log(`Transformer layers: ${options.numLayers}`);
  console.log('');

  // 語彙を作成
  console.log('Creating vocabulary...');
  const vocab = createVocab(trainingData);
  console.log(`Vocabulary size: ${vocab.length} words\n`);

  // モデルを作成
  console.log('Initializing model...');
  const llm = new SimpleLLM(vocab, options.embeddingDim, options.numLayers);
  console.log('Model initialized\n');

  // トレーニング開始
  console.log('Starting training...');
  console.log('---');

  const startTime = Date.now();
  llm.train(trainingData, options.epochs);
  const endTime = Date.now();

  console.log('---');
  console.log(`\n✓ Training completed in ${((endTime - startTime) / 1000).toFixed(2)} seconds`);

  return llm;
}

function saveModel(llm: SimpleLLM, modelName: string, metadata?: any): string {
  const modelsDir = path.join(process.cwd(), 'models');

  // modelsディレクトリを作成
  if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir, { recursive: true });
    console.log(`Created models directory: ${modelsDir}`);
  }

  // モデルをシリアライズ
  const serialized = llm.serialize();

  // メタデータを追加
  const modelData = {
    ...serialized,
    metadata: {
      name: modelName,
      createdAt: new Date().toISOString(),
      trainingTime: metadata?.trainingTime || 0,
      trainingSamples: metadata?.trainingSamples || 0,
      ...metadata,
    },
  };

  // ファイルパス
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);

  // バイナリフォーマット（MessagePack）で保存
  const binaryFileName = `${modelName}-${timestamp}.msgpack`;
  const binaryFilePath = path.join(modelsDir, binaryFileName);
  const binaryLatestPath = path.join(modelsDir, `${modelName}-latest.msgpack`);

  const binaryData = llm.serializeBinary();
  fs.writeFileSync(binaryFilePath, binaryData);
  console.log(`\n✓ Model saved to: ${binaryFilePath}`);

  // 最新版を上書き保存（バイナリ）
  fs.writeFileSync(binaryLatestPath, binaryData);
  console.log(`✓ Latest model saved to: ${binaryLatestPath}`);

  const binaryFileSize = (fs.statSync(binaryFilePath).size / 1024).toFixed(2);
  console.log(`✓ Binary file size: ${binaryFileSize} KB`);

  // JSON形式でも保存（比較・デバッグ用）
  const jsonFileName = `${modelName}-${timestamp}.json`;
  const jsonFilePath = path.join(modelsDir, jsonFileName);
  fs.writeFileSync(jsonFilePath, JSON.stringify(modelData, null, 2), 'utf-8');
  const jsonFileSize = (fs.statSync(jsonFilePath).size / 1024).toFixed(2);
  console.log(`✓ JSON file size: ${jsonFileSize} KB (${((1 - parseFloat(binaryFileSize) / parseFloat(jsonFileSize)) * 100).toFixed(1)}% reduction with binary)`);

  return binaryFilePath;
}

function testModel(llm: SimpleLLM, testInputs: string[] = ['hello', 'how are you', 'what is']) {
  console.log('\n=== Model Testing ===\n');

  testInputs.forEach(input => {
    console.log(`Input: "${input}"`);
    const output = llm.predict(input, 10);
    console.log(`Output: "${output}"`);
    console.log('---');
  });
}

function parseArguments(): {
  trainingDataFile: string;
  options: TrainingOptions;
} {
  const args = process.argv.slice(2);

  if (args.length < 1) {
    console.log('Usage: npx ts-node scripts/train-model.ts <training-data-file> [model-name] [options]');
    console.log('\nOptions:');
    console.log('  --epochs <n>      Number of training epochs (default: 50)');
    console.log('  --embedding <n>   Embedding dimension (default: 32)');
    console.log('  --layers <n>      Number of transformer layers (default: 2)');
    console.log('  --test            Run test predictions after training');
    console.log('\nExamples:');
    console.log('  npx ts-node scripts/train-model.ts data/training-data.json');
    console.log('  npx ts-node scripts/train-model.ts data/training-data.json my-model --epochs 100');
    console.log('  npx ts-node scripts/train-model.ts data/training-data.json my-model --embedding 64 --layers 3 --test');
    process.exit(1);
  }

  const trainingDataFile = args[0];
  const modelName = args[1] && !args[1].startsWith('--') ? args[1] : 'model';

  const options: TrainingOptions = {
    epochs: 50,
    embeddingDim: 32,
    numLayers: 2,
    modelName,
  };

  // オプションをパース
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--epochs' && args[i + 1]) {
      options.epochs = parseInt(args[i + 1], 10);
    } else if (args[i] === '--embedding' && args[i + 1]) {
      options.embeddingDim = parseInt(args[i + 1], 10);
    } else if (args[i] === '--layers' && args[i + 1]) {
      options.numLayers = parseInt(args[i + 1], 10);
    }
  }

  return { trainingDataFile, options };
}

// スクリプトとして実行された場合
if (require.main === module) {
  try {
    const { trainingDataFile, options } = parseArguments();

    // トレーニングデータをロード
    const trainingData = loadTrainingData(trainingDataFile);

    // モデルをトレーニング
    const startTime = Date.now();
    const llm = trainModel(trainingData, options);
    const endTime = Date.now();

    // モデルを保存
    const metadata = {
      trainingTime: (endTime - startTime) / 1000,
      trainingSamples: trainingData.length,
      epochs: options.epochs,
      embeddingDim: options.embeddingDim,
      numLayers: options.numLayers,
    };
    saveModel(llm, options.modelName, metadata);

    // テスト実行
    if (process.argv.includes('--test')) {
      testModel(llm);
    }

    console.log('\n✓ All done!');
  } catch (error) {
    console.error('Fatal error:', error);
    process.exit(1);
  }
}

export { trainModel, saveModel, testModel, loadTrainingData };
