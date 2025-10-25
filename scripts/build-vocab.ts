#!/usr/bin/env node
/**
 * Vocabulary Builder Script
 *
 * This script builds a vocabulary from a text corpus by:
 * 1. Reading text files from specified paths
 * 2. Tokenizing and counting word frequencies
 * 3. Selecting top N most frequent words
 * 4. Adding special tokens ([PAD], [UNK], [EOS], [BOS])
 * 5. Saving vocabulary to a JSON file
 *
 * Usage:
 *   npm run build-vocab -- --input corpus.txt --output vocab.json --size 5000
 */

import * as fs from 'fs';
import * as path from 'path';

interface VocabBuilderOptions {
  inputPaths: string[];
  outputPath: string;
  vocabSize: number;
  minFrequency: number;
  caseSensitive: boolean;
  specialTokens: string[];
}

class VocabularyBuilder {
  private wordCounts: Map<string, number> = new Map();
  private options: VocabBuilderOptions;

  constructor(options: Partial<VocabBuilderOptions> = {}) {
    this.options = {
      inputPaths: options.inputPaths || ['./sample.txt'],
      outputPath: options.outputPath || './data/vocab.json',
      vocabSize: options.vocabSize || 5000,
      minFrequency: options.minFrequency || 1,
      caseSensitive: options.caseSensitive || false,
      specialTokens: options.specialTokens || ['[PAD]', '[UNK]', '[EOS]', '[BOS]'],
    };
  }

  /**
   * Read and process text from input files
   */
  private readCorpus(filePath: string): string {
    console.log(`Reading corpus from: ${filePath}`);
    if (!fs.existsSync(filePath)) {
      throw new Error(`File not found: ${filePath}`);
    }
    return fs.readFileSync(filePath, 'utf-8');
  }

  /**
   * Tokenize text into words
   */
  private tokenize(text: string): string[] {
    // Normalize text
    let normalized = text;
    if (!this.options.caseSensitive) {
      normalized = normalized.toLowerCase();
    }

    // Simple word tokenization: split by whitespace and punctuation
    // Keep periods, commas, etc. as separate tokens for better language modeling
    const tokens = normalized
      .replace(/([.,!?;:])/g, ' $1 ')  // Add spaces around punctuation
      .split(/\s+/)
      .map(t => t.trim())
      .filter(t => t.length > 0);

    return tokens;
  }

  /**
   * Count word frequencies in the corpus
   */
  private countWords(texts: string[]): void {
    console.log('Counting word frequencies...');
    for (const text of texts) {
      const tokens = this.tokenize(text);
      for (const token of tokens) {
        this.wordCounts.set(token, (this.wordCounts.get(token) || 0) + 1);
      }
    }
    console.log(`Total unique words: ${this.wordCounts.size}`);
    console.log(`Total tokens: ${Array.from(this.wordCounts.values()).reduce((a, b) => a + b, 0)}`);
  }

  /**
   * Select top N most frequent words
   */
  private selectTopWords(): string[] {
    console.log(`Selecting top ${this.options.vocabSize} words...`);

    // Filter by minimum frequency
    const filtered = Array.from(this.wordCounts.entries())
      .filter(([_, count]) => count >= this.options.minFrequency);

    // Sort by frequency (descending)
    const sorted = filtered.sort((a, b) => b[1] - a[1]);

    // Take top N words (excluding special tokens count)
    const numRegularWords = this.options.vocabSize - this.options.specialTokens.length;
    const topWords = sorted.slice(0, numRegularWords).map(([word, _]) => word);

    return topWords;
  }

  /**
   * Build vocabulary with special tokens
   */
  private buildVocabulary(words: string[]): string[] {
    // Special tokens come first
    const vocab = [...this.options.specialTokens, ...words];

    console.log(`\nVocabulary built:`);
    console.log(`  Special tokens: ${this.options.specialTokens.length}`);
    console.log(`  Regular words: ${words.length}`);
    console.log(`  Total vocabulary size: ${vocab.length}`);

    return vocab;
  }

  /**
   * Save vocabulary to JSON file
   */
  private saveVocabulary(vocab: string[]): void {
    const outputDir = path.dirname(this.options.outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    fs.writeFileSync(this.options.outputPath, JSON.stringify(vocab, null, 2));
    console.log(`\nVocabulary saved to: ${this.options.outputPath}`);

    // Also save statistics
    const statsPath = this.options.outputPath.replace('.json', '-stats.json');
    const stats = {
      totalVocabSize: vocab.length,
      specialTokens: this.options.specialTokens.length,
      regularWords: vocab.length - this.options.specialTokens.length,
      totalUniqueWords: this.wordCounts.size,
      topWords: vocab.slice(this.options.specialTokens.length, this.options.specialTokens.length + 20)
        .map(word => ({
          word,
          frequency: this.wordCounts.get(word) || 0,
        })),
    };
    fs.writeFileSync(statsPath, JSON.stringify(stats, null, 2));
    console.log(`Statistics saved to: ${statsPath}`);
  }

  /**
   * Build vocabulary from corpus
   */
  build(): void {
    console.log('=== Vocabulary Builder ===\n');
    console.log('Options:');
    console.log(`  Input paths: ${this.options.inputPaths.join(', ')}`);
    console.log(`  Output path: ${this.options.outputPath}`);
    console.log(`  Target vocab size: ${this.options.vocabSize}`);
    console.log(`  Min frequency: ${this.options.minFrequency}`);
    console.log(`  Case sensitive: ${this.options.caseSensitive}`);
    console.log(`  Special tokens: ${this.options.specialTokens.join(', ')}`);
    console.log();

    // Read all input files
    const texts = this.options.inputPaths.map(p => this.readCorpus(p));

    // Count word frequencies
    this.countWords(texts);

    // Select top words
    const topWords = this.selectTopWords();

    // Build vocabulary
    const vocab = this.buildVocabulary(topWords);

    // Save to file
    this.saveVocabulary(vocab);

    console.log('\n✓ Vocabulary building completed successfully!');
  }
}

// CLI Interface
function parseArgs(): Partial<VocabBuilderOptions> {
  const args = process.argv.slice(2);
  const options: Partial<VocabBuilderOptions> = {};

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--input':
      case '-i':
        options.inputPaths = args[++i].split(',').map(p => p.trim());
        break;
      case '--output':
      case '-o':
        options.outputPath = args[++i];
        break;
      case '--size':
      case '-s':
        options.vocabSize = parseInt(args[++i], 10);
        break;
      case '--min-frequency':
      case '-f':
        options.minFrequency = parseInt(args[++i], 10);
        break;
      case '--case-sensitive':
        options.caseSensitive = true;
        break;
      case '--help':
      case '-h':
        console.log(`
Vocabulary Builder Script

Usage:
  npm run build-vocab -- [options]

Options:
  -i, --input <paths>        Comma-separated input file paths (default: ./sample.txt)
  -o, --output <path>        Output vocabulary file path (default: ./data/vocab.json)
  -s, --size <number>        Target vocabulary size (default: 5000)
  -f, --min-frequency <n>    Minimum word frequency (default: 1)
  --case-sensitive           Enable case-sensitive tokenization (default: false)
  -h, --help                 Show this help message

Examples:
  npm run build-vocab -- --input corpus.txt --size 10000
  npm run build-vocab -- --input "file1.txt,file2.txt" --output vocab.json
        `);
        process.exit(0);
    }
  }

  return options;
}

// Main execution
if (require.main === module) {
  try {
    const options = parseArgs();
    const builder = new VocabularyBuilder(options);
    builder.build();
  } catch (error) {
    console.error('Error:', error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

export { VocabularyBuilder, VocabBuilderOptions };
