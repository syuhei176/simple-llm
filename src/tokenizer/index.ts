// 簡易的なTokenizer
export class SimpleTokenizer {
  vocab: string[];
  vocabMap: Record<string, number>;

  constructor(vocabList: string[]) {
    this.vocab = vocabList;
    this.vocabMap = Object.fromEntries(vocabList.map((word, idx) => [word, idx]));
  }

  encode(text: string): number[] {
    return text.split(' ').map(w => this.vocabMap[w] ?? this.vocabMap['[UNK]']);
  }

  decode(tokens: number[]): string {
    return tokens.map(t => this.vocab[t] ?? '[UNK]').join(' ');
  }
}