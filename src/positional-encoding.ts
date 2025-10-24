// Positional Encoding
// トークンの位置情報をエンコーディングに追加する

/**
 * 位置エンコーディングを生成
 * PE(pos, 2i) = sin(pos / 10000^(2i/d))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
 */
export function getPositionalEncoding(position: number, embeddingDim: number): number[] {
  const encoding = new Array(embeddingDim);

  for (let i = 0; i < embeddingDim; i++) {
    const angle = position / Math.pow(10000, (2 * Math.floor(i / 2)) / embeddingDim);

    if (i % 2 === 0) {
      encoding[i] = Math.sin(angle);
    } else {
      encoding[i] = Math.cos(angle);
    }
  }

  return encoding;
}

/**
 * Embeddingに位置情報を追加
 */
export function addPositionalEncoding(embeddings: number[][], embeddingDim: number): number[][] {
  return embeddings.map((embedding, position) => {
    const posEncoding = getPositionalEncoding(position, embeddingDim);
    return embedding.map((val, i) => val + posEncoding[i]);
  });
}

/**
 * 位置エンコーディングのキャッシュ（パフォーマンス最適化用）
 */
export class PositionalEncodingCache {
  private cache: Map<string, number[]> = new Map();
  private embeddingDim: number;

  constructor(embeddingDim: number) {
    this.embeddingDim = embeddingDim;
  }

  get(position: number): number[] {
    const key = `${position}`;

    if (!this.cache.has(key)) {
      const encoding = getPositionalEncoding(position, this.embeddingDim);
      this.cache.set(key, encoding);
    }

    return this.cache.get(key)!;
  }

  addToEmbedding(embedding: number[], position: number): number[] {
    const posEncoding = this.get(position);
    return embedding.map((val, i) => val + posEncoding[i]);
  }

  addToEmbeddings(embeddings: number[][]): number[][] {
    return embeddings.map((embedding, position) =>
      this.addToEmbedding(embedding, position)
    );
  }
}
