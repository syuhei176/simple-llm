// 簡易的なTokenizer
export class SimpleTokenizer {
  vocab: string[];
  vocabMap: Record<string, number>;
  unkIndex: number;

  constructor(vocabList: string[]) {
    this.vocab = vocabList;
    this.vocabMap = Object.fromEntries(vocabList.map((word, idx) => [word, idx]));
    // [UNK]トークンのインデックスを保存
    this.unkIndex = this.vocabMap['[UNK]'] ?? 0;
  }

  encode(text: string): number[] {
    // 小文字に正規化してトークン化
    return text.toLowerCase().split(' ')
      .map(w => w.trim())
      .filter(w => w.length > 0)
      .map(w => this.vocabMap[w] ?? this.unkIndex);
  }

  decode(tokens: number[]): string {
    return tokens.map(t => this.vocab[t] ?? '[UNK]').join(' ');
  }
}