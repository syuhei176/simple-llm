/**
 * 既存のJSONモデルをバイナリフォーマットに変換するスクリプト
 *
 * 使用方法:
 * npx ts-node scripts/convert-to-binary.ts <json-model-file>
 */

import * as fs from 'fs';
import * as path from 'path';
import { SimpleLLM } from '../src/llm';

function convertModelToBinary(jsonFilePath: string): void {
  console.log(`Converting ${jsonFilePath} to binary format...`);

  // JSONファイルを読み込み
  if (!fs.existsSync(jsonFilePath)) {
    console.error(`Error: File not found: ${jsonFilePath}`);
    process.exit(1);
  }

  const modelData = JSON.parse(fs.readFileSync(jsonFilePath, 'utf-8'));

  // モデルをデシリアライズ
  const llm = SimpleLLM.deserialize(modelData);

  // バイナリでシリアライズ
  const binaryData = llm.serializeBinary();

  // 出力ファイルパスを生成（.jsonを.msgpackに置換）
  const outputPath = jsonFilePath.replace('.json', '.msgpack');

  // バイナリファイルに保存
  fs.writeFileSync(outputPath, binaryData);

  // ファイルサイズを比較
  const jsonSize = fs.statSync(jsonFilePath).size;
  const binarySize = fs.statSync(outputPath).size;
  const reduction = ((1 - binarySize / jsonSize) * 100).toFixed(1);

  console.log(`\n✓ Conversion completed!`);
  console.log(`✓ JSON file: ${(jsonSize / 1024).toFixed(2)} KB`);
  console.log(`✓ Binary file: ${(binarySize / 1024).toFixed(2)} KB`);
  console.log(`✓ Size reduction: ${reduction}%`);
  console.log(`✓ Output: ${outputPath}`);
}

// コマンドライン引数をパース
if (require.main === module) {
  const args = process.argv.slice(2);

  if (args.length < 1) {
    console.log('Usage: npx ts-node scripts/convert-to-binary.ts <json-model-file>');
    console.log('\nExample:');
    console.log('  npx ts-node scripts/convert-to-binary.ts models/default-latest.json');
    process.exit(1);
  }

  const jsonFilePath = args[0];
  convertModelToBinary(jsonFilePath);
}
