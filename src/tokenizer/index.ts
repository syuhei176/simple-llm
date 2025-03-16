// 簡易的なTokenizer
export class SimpleTokenizer {
  vocab: Record<string, number> = {};
  invVocab: Record<number, string> = {};

  constructor(vocabList: string[]) {
    vocabList.forEach((word, idx) => {
      this.vocab[word] = idx;
      this.invVocab[idx] = word;
    });
  }

  encode(text: string): number[] {
    return text.split(' ').map(w => this.vocab[w] ?? this.vocab['[UNK]']);
  }

  decode(tokens: number[]): string {
    return tokens.map(t => this.invVocab[t] || '[UNK]').join(' ');
  }
}