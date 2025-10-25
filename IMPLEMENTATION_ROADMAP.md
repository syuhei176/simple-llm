# 小規模実用モデル実装ロードマップ

## 目標

**現在**: 36K parameters (400 vocab, 32 dim, 2 layers, 1 head)
**目標**: 1M-3M parameters (5,000-10,000 vocab, 128-256 dim, 4-8 layers, 4-8 heads)
**スケール**: 約30-80倍

## フェーズ0: アーキテクチャ改善（WASM移行前）

> **期間**: 1-2週間
> **目標**: マルチヘッドアテンション、大規模語彙、最適化手法の実装
> **効果**: モデルサイズ 36K → 500K-1M (14-28倍)

### タスク一覧

#### 0.1 マルチヘッドアテンション実装
- [ ] 0.1.1 マルチヘッドアテンション用のクラス設計
  - ファイル: `src/multi-head-attention/index.ts`
  - 単一ヘッドから複数ヘッド（4-8ヘッド）に拡張
  - ヘッドごとに独立したQ/K/V投影

- [ ] 0.1.2 アテンションヘッドの並列計算実装
  - 各ヘッドで独立した計算
  - 最後に結合（concatenate）

- [ ] 0.1.3 出力投影層の追加
  - 結合後の出力を元の次元に投影
  - `W_o: (num_heads × head_dim) → embedding_dim`

- [ ] 0.1.4 Transformerクラスの更新
  - `SimpleTransformer`を`MultiHeadTransformer`に置き換え
  - 既存の単一ヘッドとの互換性維持

- [ ] 0.1.5 マルチヘッドアテンションのテスト
  - ユニットテスト作成
  - 順伝播・逆伝播の検証

#### 0.2 語彙サイズの拡張

- [ ] 0.2.1 大規模コーパスの準備
  - 5,000-10,000単語のデータセット収集
  - 既存の`sample.txt`を拡張、または新規作成

- [ ] 0.2.2 トークナイザーの改善
  - 頻度ベースの語彙選択
  - 特殊トークン（[PAD], [UNK], [EOS], [BOS]）の適切な処理

- [ ] 0.2.3 語彙構築スクリプトの作成
  - `scripts/build-vocab.ts`
  - コーパスから上位N単語を抽出
  - 語彙ファイル（vocab.json）の生成

- [ ] 0.2.4 埋め込み層のメモリ最適化
  - 大規模語彙でのメモリ効率を検証
  - 初期化方法の改善（Xavier, He初期化）

#### 0.3 Dropout実装

- [ ] 0.3.1 Dropoutクラスの作成
  - ファイル: `src/dropout/index.ts`
  - 順伝播: ランダムにニューロンをドロップ（p=0.1）
  - 逆伝播: ドロップしたニューロンの勾配をマスク

- [ ] 0.3.2 Transformerへの統合
  - アテンション後にDropout
  - Feed-forward後にDropout
  - 学習時のみ有効、推論時は無効

- [ ] 0.3.3 Dropoutのテスト
  - 学習時・推論時の動作確認
  - 過学習抑制効果の検証

#### 0.4 Adamオプティマイザ実装

- [ ] 0.4.1 Optimizerインターフェースの定義
  - ファイル: `src/optimizer/index.ts`
  - `Optimizer`基底クラス
  - `SGD`, `Adam`の実装

- [ ] 0.4.2 Adam実装
  - モーメンタム（beta1=0.9）
  - 二次モーメント（beta2=0.999）
  - バイアス補正
  - 学習率スケジューリング

- [ ] 0.4.3 既存コードへの統合
  - `SimpleLLM.train()`でオプティマイザを選択可能に
  - `updateParameters()`をオプティマイザに委譲

- [ ] 0.4.4 学習曲線の比較
  - SGD vs Adam
  - 収束速度の検証

#### 0.5 学習率スケジューリング

- [ ] 0.5.1 学習率スケジューラの実装
  - ファイル: `src/scheduler/index.ts`
  - Warmup + Cosine decay
  - Linear decay

- [ ] 0.5.2 トレーニングループへの統合
  - エポックごとに学習率を更新
  - ログ出力

#### 0.6 モデル設定の拡張

- [ ] 0.6.1 設定ファイルの作成
  - `configs/small-model.json`
  - `configs/medium-model.json`
  - `configs/large-model.json`

```json
// configs/small-model.json
{
  "vocabSize": 5000,
  "embeddingDim": 128,
  "numLayers": 4,
  "numHeads": 4,
  "ffnHiddenDim": 512,
  "dropout": 0.1,
  "maxSeqLen": 256
}
```

- [ ] 0.6.2 設定ファイルからのモデル構築
  - `SimpleLLM.fromConfig(config)`メソッド
  - CLIで設定ファイルを指定可能に

#### 0.7 学習パイプラインの改善

- [ ] 0.7.1 データローダーの実装
  - バッチ処理
  - シャッフル
  - パディング処理

- [ ] 0.7.2 検証データセットの分離
  - Train/Validation split (80/20)
  - 検証損失のモニタリング
  - Early stopping

- [ ] 0.7.3 ログ・メトリクスの改善
  - 学習曲線の可視化準備（データ出力）
  - Perplexity計算
  - 定期的なチェックポイント保存

- [ ] 0.7.4 学習スクリプトの強化
  - `scripts/train-model.ts`の更新
  - コマンドライン引数の拡充
  - プログレスバー表示

#### 0.8 テストとベンチマーク

- [ ] 0.8.1 ユニットテストの作成
  - 各コンポーネントのテスト
  - `tests/`ディレクトリの整備

- [ ] 0.8.2 統合テスト
  - エンドツーエンドの学習・推論テスト
  - 異なる設定でのモデルテスト

- [ ] 0.8.3 パフォーマンスベンチマーク
  - 順伝播・逆伝播の速度計測
  - メモリ使用量の計測
  - ベースライン（現在の実装）との比較

---

## フェーズ1: WASM移行

> **期間**: 1-2週間
> **目標**: 計算速度10-15倍改善
> **効果**: モデルサイズ 1M → 2M-5M (2-5倍)

### タスク一覧

#### 1.1 開発環境セットアップ

- [ ] 1.1.1 Emscriptenのインストール
  ```bash
  # Emscripten SDK
  git clone https://github.com/emscripten-core/emsdk.git
  cd emsdk
  ./emsdk install latest
  ./emsdk activate latest
  source ./emsdk_env.sh
  ```

- [ ] 1.1.2 ビルドシステムの構築
  - `wasm/`ディレクトリ作成
  - `wasm/Makefile`または`wasm/build.sh`
  - TypeScriptビルドとの統合

- [ ] 1.1.3 WASMビルドのnpmスクリプト追加
  ```json
  "scripts": {
    "build:wasm": "cd wasm && make",
    "build:all": "npm run build:wasm && npm run build"
  }
  ```

#### 1.2 基本的な行列演算のWASM実装

- [ ] 1.2.1 行列積の実装
  - ファイル: `wasm/matmul.c`
  - `matmul_f32(A, B, C, m, n, k)`
  - キャッシュブロッキング最適化

- [ ] 1.2.2 ベクトル演算の実装
  - `vector_add(a, b, result, n)`
  - `vector_mul(a, b, result, n)`
  - `dot_product(a, b, n)`

- [ ] 1.2.3 活性化関数の実装
  - `relu(x, n)` / `relu_backward(x, grad, n)`
  - `softmax(x, n)`
  - `layer_norm(x, gamma, beta, n)`

- [ ] 1.2.4 WASMモジュールのコンパイル
  ```bash
  emcc matmul.c -O3 -s WASM=1 \
    -s EXPORTED_FUNCTIONS='["_matmul_f32"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
    -o matmul.js
  ```

#### 1.3 TypeScriptバインディング

- [ ] 1.3.1 WASMローダーの作成
  - ファイル: `src/wasm/loader.ts`
  - WASMモジュールの読み込み
  - メモリ管理（malloc/free）

- [ ] 1.3.2 ラッパークラスの作成
  - ファイル: `src/wasm/operations.ts`
  - TypeScript → WASM のブリッジ
  - Float32Arrayとの相互変換

```typescript
// 使用例
import { WasmOps } from './wasm/operations';

const wasmOps = await WasmOps.init();
const result = wasmOps.matmul(matrixA, matrixB, m, n, k);
```

- [ ] 1.3.3 エラーハンドリング
  - WASMモジュールが利用不可の場合のフォールバック
  - メモリ不足の処理

#### 1.4 Float32Arrayへの移行

- [ ] 1.4.1 Transformerクラスの更新
  - `number[][]` → `Float32Array`
  - 重み行列をフラット配列で管理
  - インデックス計算の実装

```typescript
// 変更前
wq: number[][];

// 変更後
wq: Float32Array;  // size = embeddingDim * embeddingDim

// アクセス
// wq[i][j] → wq[i * embeddingDim + j]
```

- [ ] 1.4.2 Embedding層の更新
  - Float32Arrayで管理

- [ ] 1.4.3 Output層の更新
  - Float32Arrayで管理

- [ ] 1.4.4 全ての演算をFloat32Arrayベースに変更
  - 一時バッファの管理
  - メモリ再利用の最適化

#### 1.5 WASM演算の統合

- [ ] 1.5.1 Transformerの順伝播をWASM化
  - アテンション計算
  - Feed-forward計算
  - 残差接続・LayerNorm

- [ ] 1.5.2 Transformerの逆伝播をWASM化
  - 勾配計算
  - パラメータ更新

- [ ] 1.5.3 パフォーマンステスト
  - JavaScript版との速度比較
  - 目標: 10倍以上の高速化

#### 1.6 バイナリシリアライゼーション

- [ ] 1.6.1 バイナリフォーマットの設計
  - ファイル: `docs/binary-format.md`
  - ヘッダー定義（マジックナンバー、バージョン、設定）
  - セクション定義（語彙、重み、メタデータ）

```
Binary Format:
[Header: 16 bytes]
  - Magic Number: 4 bytes (0x534C4C4D = "SLLM")
  - Version: 4 bytes
  - VocabSize: 4 bytes
  - EmbeddingDim: 4 bytes

[Vocab Section]
  - Vocab Length: 4 bytes
  - Vocab Data: UTF-8 JSON (variable)

[Weights Section]
  - Embedding: vocabSize × embeddingDim floats
  - Transformers: (per layer)
    - wq, wk, wv, w1, w2: Float32Arrays
    - layerNorm params
  - Output: embeddingDim × vocabSize floats
```

- [ ] 1.6.2 シリアライザの実装
  - ファイル: `src/serializer/binary.ts`
  - `serializeToBinary(model): ArrayBuffer`
  - 各セクションの書き込み

- [ ] 1.6.3 デシリアライザの実装
  - `deserializeFromBinary(buffer): SimpleLLM`
  - ヘッダー検証
  - 各セクションの読み込み

- [ ] 1.6.4 ファイル保存・読み込み
  - Node.js: `fs.writeFile` / `fs.readFile`
  - ブラウザ: IndexedDB（既存の`ModelStorage`を拡張）

- [ ] 1.6.5 互換性テスト
  - JSON形式との相互変換
  - サイズ比較（期待: 8倍削減）

---

## フェーズ2: SIMD最適化

> **期間**: 1週間
> **目標**: さらに2倍高速化
> **効果**: モデルサイズ 5M → 10M (2倍)

### タスク一覧

#### 2.1 WASM SIMD実装

- [ ] 2.1.1 SIMD版行列積の実装
  - ファイル: `wasm/matmul_simd.c`
  - `#include <wasm_simd128.h>`
  - 4要素並列処理

- [ ] 2.1.2 SIMD版ベクトル演算
  - ベクトル加算/乗算
  - ドット積

- [ ] 2.1.3 SIMD対応のビルド
  ```bash
  emcc matmul_simd.c -O3 -msimd128 \
    -s WASM=1 -o matmul_simd.js
  ```

- [ ] 2.1.4 実行時SIMD検出
  - SIMDサポートの判定
  - フォールバック実装

- [ ] 2.1.5 パフォーマンステスト
  - SIMD vs 非SIMD
  - 目標: 2倍以上の高速化

---

## フェーズ3: 量子化（オプション）

> **期間**: 1-2週間
> **目標**: メモリ効率4倍改善
> **効果**: モデルサイズ 10M → 30M (3倍、メモリ制約緩和により)

### タスク一覧

#### 3.1 動的量子化（推論時のみ）

- [ ] 3.1.1 Q8量子化の実装
  - FP32 → INT8 変換
  - スケール・ゼロポイントの計算

- [ ] 3.1.2 量子化対応の行列積
  - INT8 × INT8 → INT32 → FP32
  - WASM実装

- [ ] 3.1.3 精度テスト
  - FP32との比較
  - 許容誤差の検証

#### 3.2 Q4量子化（上級）

- [ ] 3.2.1 4ビット量子化の実装
  - ブロック単位での量子化
  - スケール保存（FP16）

- [ ] 3.2.2 量子化モデルのシリアライゼーション
  - バイナリフォーマット拡張
  - 量子化タイプの保存

---

## フェーズ4: 並列化（オプション）

> **期間**: 1-2週間
> **目標**: マルチコア活用
> **効果**: 学習速度4倍（4コア時）

### タスク一覧

#### 4.1 Web Workers実装

- [ ] 4.1.1 Workerスクリプトの作成
  - ファイル: `src/worker/llm-worker.ts`
  - メッセージハンドラ

- [ ] 4.1.2 SharedArrayBufferの活用
  - 重みの共有
  - COOP/COEP設定

- [ ] 4.1.3 バッチ並列処理
  - トレーニングデータを分割
  - 勾配集約

- [ ] 4.1.4 パフォーマンステスト
  - 1コア vs 4コア
  - オーバーヘッドの測定

---

## マイルストーン

### M0: アーキテクチャ改善完了（2週間後）
- ✅ マルチヘッドアテンション
- ✅ 語彙5,000単語
- ✅ Adam optimizer
- ✅ Dropout
- ✅ モデルサイズ: 500K-1M

### M1: WASM移行完了（4週間後）
- ✅ WASM演算
- ✅ Float32Array
- ✅ バイナリシリアライゼーション
- ✅ 10倍高速化
- ✅ モデルサイズ: 2M-5M

### M2: SIMD最適化完了（5週間後）
- ✅ SIMD演算
- ✅ 20倍高速化（累計）
- ✅ モデルサイズ: 5M-10M

### M3: 量子化完了（7週間後、オプション）
- ✅ Q8量子化
- ✅ メモリ効率4倍
- ✅ モデルサイズ: 10M-30M

### M4: 並列化完了（9週間後、オプション）
- ✅ Web Workers
- ✅ 学習速度4倍
- ✅ 最終目標達成

---

## 実装優先順位

### 最優先（フェーズ0）
1. マルチヘッドアテンション（0.1）
2. 語彙サイズ拡張（0.2）
3. Adam optimizer（0.4）
4. モデル設定の拡張（0.6）

### 高優先（フェーズ1）
1. WASM基本セットアップ（1.1, 1.2）
2. Float32Array移行（1.4）
3. バイナリシリアライゼーション（1.6）

### 中優先（フェーズ2）
1. SIMD最適化（2.1）

### 低優先（オプション）
1. Dropout（0.3）- 過学習が問題になったら
2. 量子化（フェーズ3）- メモリが足りなくなったら
3. 並列化（フェーズ4）- 学習時間が問題になったら

---

## 開発ワークフロー

各タスクの実装手順:
1. ✅ ブランチ作成: `git checkout -b feature/task-name`
2. ✅ 実装
3. ✅ テスト作成・実行
4. ✅ ドキュメント更新
5. ✅ コミット: `git commit -m "Implement task-name"`
6. ✅ マージ: `git merge feature/task-name`
7. ✅ タグ付け（マイルストーン時）: `git tag -a v0.1.0 -m "Milestone M0"`

---

## 成功指標

### 技術指標
- ✅ モデルサイズ: 1M-10M parameters
- ✅ 計算速度: 20倍以上高速化
- ✅ メモリ効率: 4倍以上改善
- ✅ ファイルサイズ: 8倍以上削減

### 品質指標
- ✅ ユニットテストカバレッジ: 70%以上
- ✅ ドキュメント: 全公開APIにJSDoc
- ✅ ベンチマーク: 各フェーズで計測

### 実用性指標
- ✅ 簡単な対話が可能
- ✅ ドメイン特化タスクで実用的
- ✅ ブラウザで学習・推論が快適に動作
