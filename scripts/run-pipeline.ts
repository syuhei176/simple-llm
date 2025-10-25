/**
 * 完全なパイプラインを実行するスクリプト
 *
 * 使用方法:
 * npx ts-node scripts/run-pipeline.ts [options]
 *
 * 例:
 * npx ts-node scripts/run-pipeline.ts --sample
 * npx ts-node scripts/run-pipeline.ts --url https://example.com/text.txt --window 7 --epochs 100
 */

import * as fs from 'fs';
import * as path from 'path';
import { fetchTextFromUrl, saveText } from './fetch-text';
import { convertToTrainingData } from './prepare-training-data';
import { trainModel, saveModel, testModel, loadTrainingData } from './train-model';

interface PipelineOptions {
  // Step 1: Fetch
  url?: string;
  useSample: boolean;
  corpusFile: string;

  // Step 2: Prepare
  trainingFile: string;
  windowSize: number;
  stride: number;
  maxSamples?: number;

  // Step 3: Train
  modelName: string;
  epochs: number;
  embeddingDim: number;
  numLayers: number;
  runTest: boolean;

  // Step 4: Deploy
  setAsDefault: boolean;
}

async function runPipeline(options: PipelineOptions) {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║         Simple LLM Training Pipeline                       ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');

  try {
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Step 1: Fetch Text
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    console.log('┌─────────────────────────────────────────────────────────┐');
    console.log('│ Step 1/4: Fetching Text                                 │');
    console.log('└─────────────────────────────────────────────────────────┘\n');

    let text: string;

    if (fs.existsSync(options.corpusFile)) {
      console.log(`Using existing corpus file: ${options.corpusFile}`);
      text = fs.readFileSync(options.corpusFile, 'utf-8');
    } else if (options.useSample) {
      console.log('Fetching sample text from Project Gutenberg...');
      const sampleUrl = 'https://mirror.csclub.uwaterloo.ca/gutenberg/3/0/2/7/30272/30272-0.txt';
      text = await fetchTextFromUrl(sampleUrl);
      saveText(text, options.corpusFile);
    } else if (options.url) {
      console.log(`Fetching text from: ${options.url}`);
      text = await fetchTextFromUrl(options.url);
      saveText(text, options.corpusFile);
    } else {
      throw new Error('No text source specified. Use --sample or --url');
    }

    console.log(`✓ Text fetched: ${text.length} characters\n`);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Step 2: Prepare Training Data
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    console.log('┌─────────────────────────────────────────────────────────┐');
    console.log('│ Step 2/4: Preparing Training Data                       │');
    console.log('└─────────────────────────────────────────────────────────┘\n');

    convertToTrainingData(options.corpusFile, options.trainingFile, {
      windowSize: options.windowSize,
      stride: options.stride,
      maxSamples: options.maxSamples,
    });

    console.log(`✓ Training data prepared\n`);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Step 3: Train Model
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    console.log('┌─────────────────────────────────────────────────────────┐');
    console.log('│ Step 3/4: Training Model                                │');
    console.log('└─────────────────────────────────────────────────────────┘\n');

    const trainingData = loadTrainingData(options.trainingFile);
    const startTime = Date.now();

    const llm = trainModel(trainingData, {
      epochs: options.epochs,
      embeddingDim: options.embeddingDim,
      numLayers: options.numLayers,
      modelName: options.modelName,
    });

    const endTime = Date.now();
    const trainingTime = (endTime - startTime) / 1000;

    const metadata = {
      trainingTime,
      trainingSamples: trainingData.length,
      epochs: options.epochs,
      embeddingDim: options.embeddingDim,
      numLayers: options.numLayers,
      source: options.url || (options.useSample ? 'sample' : 'local'),
    };

    const modelPath = saveModel(llm, options.modelName, metadata);

    console.log(`✓ Model trained and saved\n`);

    // Test model if requested
    if (options.runTest) {
      testModel(llm);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Step 4: Deploy Model
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    console.log('┌─────────────────────────────────────────────────────────┐');
    console.log('│ Step 4/4: Deploying Model                               │');
    console.log('└─────────────────────────────────────────────────────────┘\n');

    if (options.setAsDefault) {
      const latestPath = path.join(process.cwd(), 'models', `${options.modelName}-latest.json`);
      const defaultPath = path.join(process.cwd(), 'models', 'default-latest.json');

      fs.copyFileSync(latestPath, defaultPath);
      console.log(`✓ Copied ${options.modelName}-latest.json to default-latest.json`);
      console.log('  This model will be used by the WebUI automatically.\n');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Summary
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    console.log('╔════════════════════════════════════════════════════════════╗');
    console.log('║              Pipeline Completed Successfully!              ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    console.log('Summary:');
    console.log(`  Model: ${options.modelName}`);
    console.log(`  Training samples: ${trainingData.length}`);
    console.log(`  Training time: ${trainingTime.toFixed(2)}s`);
    console.log(`  Model file: ${modelPath}`);

    console.log('\nNext steps:');
    console.log('  1. Review the model file in models/ directory');
    console.log('  2. Commit the model to git: git add models/ && git commit -m "Add trained model"');
    console.log('  3. Run the WebUI: npm run dev');
    console.log('  4. Test the model in your browser: http://localhost:8080\n');

  } catch (error) {
    console.error('\n❌ Pipeline failed:', error);
    process.exit(1);
  }
}

function parseArguments(): PipelineOptions {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log('Usage: npx ts-node scripts/run-pipeline.ts [options]');
    console.log('\nOptions:');
    console.log('  --sample              Use sample text from Project Gutenberg');
    console.log('  --url <url>           Fetch text from URL');
    console.log('  --corpus <file>       Corpus file path (default: data/corpus.txt)');
    console.log('  --training <file>     Training data file path (default: data/training-data.json)');
    console.log('  --model <name>        Model name (default: default)');
    console.log('  --window <n>          Window size (default: 5)');
    console.log('  --stride <n>          Stride (default: 1)');
    console.log('  --max <n>             Maximum samples');
    console.log('  --epochs <n>          Training epochs (default: 50)');
    console.log('  --embedding <n>       Embedding dimension (default: 32)');
    console.log('  --layers <n>          Number of layers (default: 2)');
    console.log('  --test                Run test after training');
    console.log('  --set-default         Set as default model for WebUI');
    console.log('\nExamples:');
    console.log('  npx ts-node scripts/run-pipeline.ts --sample --set-default');
    console.log('  npx ts-node scripts/run-pipeline.ts --url https://example.com/text.txt --window 7 --epochs 100');
    console.log('  npx ts-node scripts/run-pipeline.ts --sample --model my-model --embedding 64 --layers 3 --test');
    process.exit(0);
  }

  const options: PipelineOptions = {
    useSample: false,
    corpusFile: 'data/corpus.txt',
    trainingFile: 'data/training-data.json',
    modelName: 'default',
    windowSize: 5,
    stride: 1,
    epochs: 50,
    embeddingDim: 32,
    numLayers: 2,
    runTest: false,
    setAsDefault: false,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--sample':
        options.useSample = true;
        break;
      case '--url':
        options.url = args[++i];
        break;
      case '--corpus':
        options.corpusFile = args[++i];
        break;
      case '--training':
        options.trainingFile = args[++i];
        break;
      case '--model':
        options.modelName = args[++i];
        break;
      case '--window':
        options.windowSize = parseInt(args[++i], 10);
        break;
      case '--stride':
        options.stride = parseInt(args[++i], 10);
        break;
      case '--max':
        options.maxSamples = parseInt(args[++i], 10);
        break;
      case '--epochs':
        options.epochs = parseInt(args[++i], 10);
        break;
      case '--embedding':
        options.embeddingDim = parseInt(args[++i], 10);
        break;
      case '--layers':
        options.numLayers = parseInt(args[++i], 10);
        break;
      case '--test':
        options.runTest = true;
        break;
      case '--set-default':
        options.setAsDefault = true;
        break;
    }
  }

  return options;
}

// スクリプトとして実行された場合
if (require.main === module) {
  const options = parseArguments();
  runPipeline(options).catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export { runPipeline };
