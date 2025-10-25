# GitHub Actions for Training Pipeline

このプロジェクトはGitHub Actionsを使用してモデルトレーニングパイプラインを自動実行できます。

## ワークフロー一覧

### 1. Train Language Model (`train-model.yml`)

完全なトレーニングパイプラインを実行します。

#### トリガー

- **手動実行** (`workflow_dispatch`): GitHubのActionsタブから実行
- **スケジュール実行**: 毎週日曜日 0:00 UTC
- **プルリクエスト**: `data/` または `scripts/` ディレクトリに変更があった場合

#### 手動実行方法

1. GitHubリポジトリの **Actions** タブを開く
2. 左サイドバーから **Train Language Model** を選択
3. **Run workflow** ボタンをクリック
4. パラメータを設定:

| パラメータ | 説明 | デフォルト |
|-----------|------|----------|
| data_source | データソース (sample/url/file) | sample |
| data_url | URLからデータを取得する場合のURL | - |
| model_name | モデル名 | default |
| window_size | スライディングウィンドウサイズ | 5 |
| stride | ウィンドウの移動幅 | 1 |
| epochs | トレーニングエポック数 | 50 |
| embedding_dim | 埋め込み次元数 | 32 |
| num_layers | Transformerレイヤー数 | 2 |
| max_samples | 最大サンプル数（オプション） | - |
| set_as_default | デフォルトモデルとして設定 | true |

5. **Run workflow** で実行開始

#### 実行内容

1. Node.js環境のセットアップ
2. 依存パッケージのインストール
3. トレーニングデータの取得
4. データの前処理と変換
5. モデルのトレーニング
6. モデルの保存
7. デフォルトモデルとして設定（オプション）
8. Webバンドルのビルド
9. トレーニングレポートの生成
10. モデルのコミット＆プッシュ
11. アーティファクトのアップロード

#### 出力

- **コミット**: 学習済みモデルが自動的にリポジトリにコミットされます
- **アーティファクト**:
  - `trained-model-{model_name}`: 学習済みモデル（30日間保持）
  - `training-data`: トレーニングデータ（7日間保持）
- **レポート**: `training-report.md` が生成されます

### 2. Quick Pipeline Test (`quick-test.yml`)

パイプラインの動作確認用の軽量テストです。

#### トリガー

- **プルリクエスト**: `scripts/`、`src/`、`.github/workflows/` に変更があった場合
- **手動実行** (`workflow_dispatch`)

#### 実行内容

1. テストコーパスの作成
2. データ準備のテスト
3. 小規模モデルのトレーニング（5エポック）
4. モデルファイルの検証
5. モデル構造の検証
6. テスト結果のアップロード

#### 実行時間

約5-10分で完了します（フルトレーニングは30-60分）

## 使用例

### 例1: サンプルデータで標準的なトレーニング

```yaml
data_source: sample
model_name: default
window_size: 5
stride: 1
epochs: 50
embedding_dim: 32
num_layers: 2
set_as_default: true
```

### 例2: URLから高品質モデルをトレーニング

```yaml
data_source: url
data_url: https://www.gutenberg.org/files/1342/1342-0.txt
model_name: shakespeare
window_size: 7
stride: 2
epochs: 100
embedding_dim: 64
num_layers: 3
max_samples: 2000
set_as_default: true
```

### 例3: 高速な小規模モデル

```yaml
data_source: sample
model_name: small-model
window_size: 3
stride: 1
epochs: 30
embedding_dim: 16
num_layers: 2
max_samples: 500
set_as_default: false
```

## トラブルシューティング

### ワークフローが失敗する場合

1. **タイムアウトエラー**
   - `max_samples` を減らす
   - `epochs` を減らす
   - `embedding_dim` や `num_layers` を小さくする

2. **メモリ不足**
   - トレーニングデータのサイズを制限
   - `max_samples` パラメータを使用

3. **コミット失敗**
   - リポジトリの権限を確認
   - GitHub Actionsの `contents: write` 権限を確認

### アーティファクトのダウンロード

1. GitHubの **Actions** タブを開く
2. 完了したワークフロー実行を選択
3. ページ下部の **Artifacts** セクションからダウンロード

## 自動化のベストプラクティス

### 1. スケジュール実行の設定

定期的に最新データで再トレーニング：

```yaml
schedule:
  - cron: '0 0 * * 0'  # 毎週日曜日
```

### 2. データソースの管理

`data/sources.txt` にURLリストを保存：

```txt
https://www.gutenberg.org/files/1342/1342-0.txt
https://example.com/corpus1.txt
https://example.com/corpus2.txt
```

### 3. モデルバージョニング

モデル名に日付を含める：

```yaml
model_name: model-${{ github.run_number }}
```

### 4. プルリクエストでの検証

新しいトレーニングデータを追加する際：

1. `data/` ディレクトリにファイルを追加
2. プルリクエストを作成
3. 自動的にトレーニングが実行される
4. 結果をレビュー
5. 問題なければマージ

## セキュリティとコスト

### 実行時間の制限

無料プランでは月2000分まで：

- Quick Test: ~5-10分
- Full Training: ~30-60分
- 推奨: 月20-40回のフルトレーニング

### プライベートデータの取り扱い

機密データを扱う場合：

1. GitHub Secrets にURLやトークンを保存
2. プライベートリポジトリを使用
3. アーティファクトの保持期間を短く設定

### 並列実行の制限

同時実行を防ぐには：

```yaml
concurrency:
  group: train-model
  cancel-in-progress: true
```

## モニタリング

### メール通知

GitHub設定でワークフロー通知を有効化：

1. Settings → Notifications
2. "Actions" を有効化

### Slack通知（オプション）

Slack Webhookを使用：

```yaml
- name: Notify Slack
  if: always()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
```

## 参考資料

- [GitHub Actions公式ドキュメント](https://docs.github.com/actions)
- [ワークフロー構文](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [使用制限](https://docs.github.com/en/actions/learn-github-actions/usage-limits-billing-and-administration)
