# Training Pipeline

完全な機械学習パイプラインの実装ドキュメント

## 概要

このプロジェクトは以下の4ステップで構成される完全な機械学習パイプラインを提供します：

1. **データ収集**: ネットから長い文章を収集
2. **データ変換**: 文章をトレーニングデータに変換
3. **モデル学習**: トレーニングを実行しモデルを生成
4. **モデル利用**: WebUIで学習済みモデルを使用

## パイプライン実行方法

### 方法1: GitHub Actions（推奨）

GitHubのUIから簡単に実行：

1. リポジトリの **Actions** タブを開く
2. **Train Language Model** を選択
3. **Run workflow** をクリック
4. パラメータを設定して実行

**メリット:**
- ☁️ クラウドで自動実行（ローカル環境不要）
- 📅 スケジュール実行可能（毎週自動トレーニング）
- 💾 モデルが自動的にリポジトリに保存
- 📊 実行ログとレポートが残る

詳細は [.github/ACTIONS.md](.github/ACTIONS.md) を参照。

### 方法2: ローカル実行（クイックスタート）

```bash
# 全ステップを一度に実行
npm run pipeline
```

### 方法3: ステップバイステップ実行

#### ステップ1: テキストデータの収集

```bash
# サンプルテキストをダウンロード（Project Gutenberg）
npx ts-node scripts/fetch-text.ts --sample data/corpus.txt

# 特定のURLからテキストをダウンロード
npx ts-node scripts/fetch-text.ts https://example.com/text.txt data/corpus.txt

# 複数のURLからテキストを収集
npx ts-node scripts/fetch-text.ts --multiple url1.txt,url2.txt data/corpus.txt
```

**出力**: `data/corpus.txt` - 収集したテキストファイル

#### ステップ2: トレーニングデータの準備

```bash
# デフォルト設定で変換（ウィンドウ5、ストライド1）
npx ts-node scripts/prepare-training-data.ts data/corpus.txt data/training-data.json

# カスタム設定で変換
npx ts-node scripts/prepare-training-data.ts data/corpus.txt data/training-data.json \
  --window 7 --stride 2 --max 1000
```

**パラメータ**:
- `--window <n>`: ウィンドウサイズ（デフォルト: 5）
- `--stride <n>`: 移動幅（デフォルト: 1）
- `--max <n>`: 最大サンプル数
- `--min-word <n>`: 最小単語長（デフォルト: 2）

**出力**: `data/training-data.json` - トレーニングデータ

#### ステップ3: モデルの学習と保存

```bash
# デフォルト設定でトレーニング（50エポック）
npx ts-node scripts/train-model.ts data/training-data.json my-model

# カスタム設定でトレーニング
npx ts-node scripts/train-model.ts data/training-data.json my-model \
  --epochs 100 --embedding 64 --layers 3 --test
```

**パラメータ**:
- `--epochs <n>`: エポック数（デフォルト: 50）
- `--embedding <n>`: 埋め込み次元（デフォルト: 32）
- `--layers <n>`: Transformerレイヤー数（デフォルト: 2）
- `--test`: 学習後にテスト実行

**出力**:
- `models/my-model-2025-01-01T12-00-00.json` - タイムスタンプ付きモデル
- `models/my-model-latest.json` - 最新版モデル

#### ステップ4: WebUIでモデルを使用

```bash
# Webサーバーを起動
npm run dev
```

ブラウザで `http://localhost:8080` を開くと、自動的に `models/default-latest.json` がロードされます。

## ディレクトリ構造

```
simple-llm/
├── scripts/               # パイプライン用スクリプト
│   ├── fetch-text.ts     # テキスト収集
│   ├── prepare-training-data.ts  # データ変換
│   └── train-model.ts    # モデル学習
├── data/                 # データファイル
│   ├── corpus.txt        # 収集したテキスト（Gitで管理しない）
│   └── training-data.json # トレーニングデータ
├── models/               # 学習済みモデル（Gitで管理）
│   ├── default-latest.json
│   └── my-model-2025-01-01T12-00-00.json
└── docs/                 # WebUI（GitHub Pages）
    └── index.html
```

## モデル管理

### モデルファイルの命名規則

- `<model-name>-<timestamp>.json`: バージョン管理されたモデル
- `<model-name>-latest.json`: 常に最新版を指す

### モデルのGit管理

モデルファイルはリポジトリで管理されます：

```bash
# モデルをコミット
git add models/my-model-latest.json
git commit -m "Add trained model: my-model"
git push
```

**注意**: 非常に大きなモデル（>10MB）は Git LFS の使用を検討してください。

### WebUIでのモデル使用

WebUIは起動時に以下の順序でモデルをロードします：

1. `models/default-latest.json` をリポジトリからロード
2. 見つからない場合はIndexedDBから前回保存したモデルをロード
3. どちらもない場合は手動でトレーニングが必要

## パイプラインのカスタマイズ

### データ収集のカスタマイズ

独自のテキストソースを使用する場合：

```typescript
// scripts/fetch-text.ts を編集
const SAMPLE_URLS = [
  'https://your-source.com/text1.txt',
  'https://your-source.com/text2.txt',
];
```

### トレーニングパラメータの推奨設定

| データサイズ | ウィンドウ | ストライド | エポック | 埋め込み | レイヤー |
|------------|-----------|----------|---------|---------|---------|
| 小（<500語）| 3-5 | 1 | 50-100 | 16-32 | 2 |
| 中（500-2000語）| 5-7 | 2 | 50-150 | 32-64 | 2-3 |
| 大（>2000語）| 7-10 | 3 | 100-200 | 64-128 | 3-4 |

## トラブルシューティング

### テキスト収集の問題

```bash
# HTTPエラーが発生する場合
# - URLが正しいか確認
# - ネットワーク接続を確認
# - CORSエラーの場合は別の方法でダウンロード
```

### メモリ不足エラー

```bash
# サンプル数を制限
npx ts-node scripts/prepare-training-data.ts data/corpus.txt data/training.json --max 1000

# または、ウィンドウサイズを小さくする
npx ts-node scripts/prepare-training-data.ts data/corpus.txt data/training.json --window 3
```

### モデルがロードされない

```bash
# ファイルが存在するか確認
ls -la models/

# モデルファイルの中身を確認
head models/default-latest.json

# WebUIのコンソールでエラーを確認
# (ブラウザの開発者ツールを開く)
```

## 実例

### 例1: シェイクスピアの文章で学習

```bash
# 1. テキストをダウンロード
npx ts-node scripts/fetch-text.ts \
  https://www.gutenberg.org/files/100/100-0.txt \
  data/shakespeare.txt

# 2. トレーニングデータに変換
npx ts-node scripts/prepare-training-data.ts \
  data/shakespeare.txt \
  data/shakespeare-training.json \
  --window 7 --stride 2

# 3. モデルを学習
npx ts-node scripts/train-model.ts \
  data/shakespeare-training.json \
  shakespeare \
  --epochs 100 --embedding 64 --layers 3 --test

# 4. WebUIで使用
# models/shakespeare-latest.json を models/default-latest.json にコピー
cp models/shakespeare-latest.json models/default-latest.json
npm run dev
```

### 例2: 技術文書で学習

```bash
# 複数のMarkdownファイルから学習
cat docs/*.md > data/tech-docs.txt

npx ts-node scripts/prepare-training-data.ts \
  data/tech-docs.txt \
  data/tech-training.json \
  --window 5 --stride 1

npx ts-node scripts/train-model.ts \
  data/tech-training.json \
  tech-model \
  --epochs 80 --embedding 48 --layers 2
```

## パフォーマンス最適化

### 学習時間の短縮

1. サンプル数を制限: `--max 500`
2. エポック数を減らす: `--epochs 30`
3. レイヤー数を減らす: `--layers 2`

### モデルサイズの削減

1. 埋め込み次元を小さく: `--embedding 16`
2. 語彙サイズを制限（頻出語のみ使用）

## 参考資料

- [スライディングウィンドウの詳細](examples/sliding-window-example.ts)
- [モデルアーキテクチャ](README.md#architecture)
- [トレーニングデータ形式](src/training-data.ts)
