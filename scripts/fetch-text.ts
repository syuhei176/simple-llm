/**
 * ネットから文章を収集するスクリプト
 *
 * 使用方法:
 * npx ts-node scripts/fetch-text.ts <URL> [output-file]
 *
 * 例:
 * npx ts-node scripts/fetch-text.ts https://example.com/article.txt data/corpus.txt
 */

import * as fs from 'fs';
import * as path from 'path';

// Node.js 18+ の fetch API を使用
async function fetchTextFromUrl(url: string): Promise<string> {
  try {
    console.log(`Fetching text from: ${url}`);
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const contentType = response.headers.get('content-type');
    console.log(`Content-Type: ${contentType}`);

    const text = await response.text();
    console.log(`Fetched ${text.length} characters`);

    // HTMLの場合は簡易的なテキスト抽出
    if (contentType?.includes('text/html')) {
      console.log('Detected HTML, extracting text content...');
      // 簡易的なHTMLタグ除去（本格的にはDOM parserが必要）
      const cleanText = text
        .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
        .replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '')
        .replace(/<[^>]+>/g, ' ')
        .replace(/&nbsp;/g, ' ')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&amp;/g, '&')
        .replace(/\s+/g, ' ')
        .trim();

      return cleanText;
    }

    return text;
  } catch (error) {
    console.error('Error fetching text:', error);
    throw error;
  }
}

async function fetchMultipleUrls(urls: string[]): Promise<string> {
  console.log(`Fetching text from ${urls.length} URLs...`);
  const texts: string[] = [];

  for (const url of urls) {
    try {
      const text = await fetchTextFromUrl(url);
      texts.push(text);
      console.log(`✓ Successfully fetched from ${url}`);
    } catch (error) {
      console.error(`✗ Failed to fetch from ${url}:`, error);
    }
  }

  return texts.join('\n\n');
}

function saveText(text: string, outputPath: string) {
  const dir = path.dirname(outputPath);

  // ディレクトリが存在しない場合は作成
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  fs.writeFileSync(outputPath, text, 'utf-8');
  console.log(`\nSaved text to: ${outputPath}`);
  console.log(`Total size: ${text.length} characters`);
  console.log(`Word count: ~${text.split(/\s+/).length} words`);
}

// サンプルテキストソース（パブリックドメインのテキスト）
const SAMPLE_URLS = [
  // Project Gutenberg の公開テキスト（パブリックドメイン）
  'https://www.gutenberg.org/files/1342/1342-0.txt', // Pride and Prejudice
  // 他のパブリックドメインソース
];

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log('Usage: npx ts-node scripts/fetch-text.ts <URL> [output-file]');
    console.log('   or: npx ts-node scripts/fetch-text.ts --sample [output-file]');
    console.log('\nExamples:');
    console.log('  npx ts-node scripts/fetch-text.ts https://example.com/text.txt');
    console.log('  npx ts-node scripts/fetch-text.ts --sample data/corpus.txt');
    console.log('  npx ts-node scripts/fetch-text.ts --multiple url1.txt,url2.txt data/output.txt');
    process.exit(1);
  }

  const outputFile = args[args.length - 1].includes('.txt')
    ? args[args.length - 1]
    : 'data/corpus.txt';

  let text: string;

  if (args[0] === '--sample') {
    console.log('Fetching sample text from Project Gutenberg...');
    text = await fetchTextFromUrl(SAMPLE_URLS[0]);
  } else if (args[0] === '--multiple') {
    const urls = args[1].split(',');
    text = await fetchMultipleUrls(urls);
  } else if (args[0].startsWith('http://') || args[0].startsWith('https://')) {
    text = await fetchTextFromUrl(args[0]);
  } else {
    console.error('Error: URL must start with http:// or https://');
    process.exit(1);
  }

  saveText(text, outputFile);
}

// スクリプトとして実行された場合
if (require.main === module) {
  main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export { fetchTextFromUrl, fetchMultipleUrls, saveText };
