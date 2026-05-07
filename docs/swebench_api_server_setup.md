# SWE-Bench APIサーバーの立ち上げとCloudflare Tunnel設定マニュアル

このマニュアルでは、SWE-Bench評価用のAPIサーバーをローカルで立ち上げ、Cloudflare Tunnelを使って外部からアクセス可能にする方法を説明します。

---

## 目次

1. [前提条件](#前提条件)
2. [依存関係のインストール](#依存関係のインストール)
3. [環境変数の設定](#環境変数の設定)
4. [APIサーバーの起動](#apiサーバーの起動)
5. [Cloudflare Tunnelの設定](#cloudflare-tunnelの設定)
6. [YAML設定ファイルの書き方](#yaml設定ファイルの書き方)
7. [動作確認](#動作確認)
8. [トラブルシューティング](#トラブルシューティング)

---

## 前提条件

以下がインストール済みであることを確認してください：

- **Docker**: SWE-Benchの評価用コンテナを実行するために必要
  ```bash
  docker --version
  ```
- **Python仮想環境**: `./myenv` ディレクトリに仮想環境が作成されていること
  ```bash
  # まだない場合は作成
  python3 -m venv myenv
  ```
- **Docker権限**: 現在のユーザーがDockerを実行できること
  ```bash
  # 権限がない場合は追加
  sudo usermod -aG docker $USER
  newgrp docker
  ```

---

## 依存関係のインストール

### 1. Python依存関係のインストール

リポジトリのルートディレクトリで以下を実行します：

```bash
cd /home/yuya/qwen3-next/llm-leaderboard

# FastAPI、Uvicorn、SWE-Benchパッケージをインストール
./myenv/bin/pip install fastapi "uvicorn[standard]" swebench python-dotenv
```

### 2. SWE-Bench Dockerイメージの準備

APIサーバーは、評価時に必要なDockerイメージを自動的にプルします。事前にプルしておく場合は：

```bash
# 例: astropy のイメージをプル
docker pull swebench/sweb.eval.x86_64.astropy__astropy-12907:latest
```

---

## 環境変数の設定

### 1. `.env` ファイルの作成

リポジトリのルートに `.env` ファイルを作成します：

```bash
cd /home/yuya/qwen3-next/llm-leaderboard
cp env.example .env
```

### 2. SWE-Bench API用の環境変数を追加

`.env` ファイルに以下を追加します：

```bash
# SWE-Bench APIサーバーのAPIキー（任意の文字列を設定）
SWE_API_KEY=your_secure_api_key_here

# ワーカー数（CPU数に応じて調整、デフォルトは4）
SWE_WORKERS=4

# サーバーポート（デフォルトは8000）
PORT=8000
```

**注意**: `SWE_API_KEY` を設定した場合、全てのAPIリクエストで `X-API-Key` ヘッダーが必須になります。

---

## APIサーバーの起動

### 方法1: 起動スクリプトを使用（推奨）

```bash
cd /home/yuya/qwen3-next/llm-leaderboard

# デフォルト設定で起動（0.0.0.0:8000）
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh

# カスタムホスト・ポートで起動
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh --host 0.0.0.0 --port 8080
```

このスクリプトは：
- 既存のサーバープロセスを自動的に停止
- `.env` ファイルを自動的に読み込み
- バックグラウンドでサーバーを起動
- ログを `/tmp/swebench_server.out` に出力

### 方法2: 直接起動

```bash
cd /home/yuya/qwen3-next/llm-leaderboard

# .env を読み込む
source .env

# バックグラウンドで起動
nohup ./myenv/bin/python scripts/server/swebench_server.py \
  --host 0.0.0.0 --port 8000 \
  >/tmp/swebench_server.out 2>&1 & disown
```

### ログの確認

```bash
# リアルタイムでログを確認
tail -f /tmp/swebench_server.out

# 最初の120行を確認
tail -f /tmp/swebench_server.out | sed -n '1,120p'
```

### サーバーの停止

```bash
# プロセスを探して停止
pkill -f swebench_server.py

# または、プロセスIDを指定
ps aux | grep swebench_server.py
kill <PID>
```

---

## Cloudflare Tunnelの設定

Cloudflare Tunnelを使うと、ローカルのAPIサーバーを固定のドメインで外部公開できます。

### 1. Cloudflare Tunnelのインストール

```bash
# Ubuntu/Debian
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# または、直接ダウンロード
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# インストール確認
cloudflared --version
```

### 2. Cloudflareアカウントでログイン

```bash
cloudflared tunnel login
```

ブラウザが開くので、Cloudflareアカウントでログインし、使用するドメインを選択します。

### 3. Tunnelの作成

```bash
# トンネルを作成（名前は任意）
cloudflared tunnel create swebench-api

# 作成されたトンネルのIDをメモしておく
# 例: Created tunnel swebench-api with id: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### 4. DNS設定

```bash
# サブドメインをトンネルにルーティング
# 例: api.nejumi-swebench.org をトンネルに接続
cloudflared tunnel route dns swebench-api api.nejumi-swebench.org
```

### 5. 設定ファイルの作成

`~/.cloudflared/config.yml` を作成します：

```yaml
tunnel: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  # 手順3で取得したトンネルID
credentials-file: /home/yuya/.cloudflared/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.json

ingress:
  # SWE-Bench APIサーバー
  - hostname: api.nejumi-swebench.org
    service: http://localhost:8000
  # その他のサービス用（必要に応じて追加）
  - service: http_status:404
```

### 6. Tunnelの起動

```bash
# フォアグラウンドで起動（テスト用）
cloudflared tunnel run swebench-api

# バックグラウンドで起動
nohup cloudflared tunnel run swebench-api >/tmp/cloudflared.log 2>&1 & disown

# ログ確認
tail -f /tmp/cloudflared.log
```

### 7. システムサービスとして登録（推奨）

恒久的に実行する場合は、systemdサービスとして登録します：

```bash
# サービスファイルをインストール
sudo cloudflared service install

# サービスを有効化・起動
sudo systemctl enable cloudflared
sudo systemctl start cloudflared

# ステータス確認
sudo systemctl status cloudflared

# ログ確認
sudo journalctl -u cloudflared -f
```

---

## YAML設定ファイルの書き方

評価時にAPIサーバーを使用するには、設定YAMLファイルで以下のように指定します。

### 例: `configs/config-gpt-4o-2024-11-20.yaml`

```yaml
model:
  name: gpt-4o-2024-11-20
  provider: openai
  
run:
  swebench: true
  # 他のベンチマークは必要に応じて設定

swebench:
  # APIサーバー経由で評価
  background_eval: true  # 非同期評価を有効化
  
  api_server:
    enabled: true
    endpoint: 'https://api.nejumi-swebench.org'  # Cloudflare Tunnelのドメイン
    api_key: null  # 環境変数 SWE_API_KEY が優先される
    
    # オプション: イメージ設定（省略可能）
    namespace: 'swebench'
    tag: 'latest'
    
    # タイムアウト設定（秒）
    timeout_sec: 1200
  
  # その他のSWE-Bench設定
  max_samples: 80
  max_workers: 24
```

### 設定パラメータの説明

- **`enabled`**: `true` でAPIサーバー経由の評価を有効化
- **`endpoint`**: APIサーバーのURL
  - ローカル: `http://127.0.0.1:8000`
  - Cloudflare Tunnel: `https://api.nejumi-swebench.org`
- **`api_key`**: APIキー（nullの場合は環境変数 `SWE_API_KEY` を使用）
- **`namespace`**: Dockerイメージの名前空間（デフォルト: `swebench`）
- **`tag`**: Dockerイメージのタグ（デフォルト: `latest`）
- **`timeout_sec`**: 評価のタイムアウト時間（秒）

---

## 動作確認

### 1. ローカルでの確認

APIサーバーが正常に起動しているか確認します：

```bash
# ヘルスチェック（APIキー不要）
curl http://localhost:8000/

# サマリー情報を取得
curl -s http://localhost:8000/v1/summary | jq
```

### 2. Cloudflare Tunnel経由での確認

```bash
# 外部からアクセス可能か確認
curl https://api.nejumi-swebench.org/

# サマリー情報を取得
curl -s https://api.nejumi-swebench.org/v1/summary | jq
```

### 3. ジョブの送信テスト

テスト用のパッチファイルを作成してジョブを送信します：

```bash
# テスト用パッチを作成
cat > test_patch.diff << 'EOF'
--- a/example.py
+++ b/example.py
@@ -1,1 +1,1 @@
-print("Hello")
+print("Hello, World!")
EOF

# ジョブを送信
INSTANCE_ID=astropy__astropy-12907
API_KEY=your_secure_api_key_here

curl -s -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d @<(jq -n \
    --arg iid "$INSTANCE_ID" \
    --arg patch "$(cat test_patch.diff)" \
    '{
      instance_id: $iid,
      patch_diff: $patch,
      namespace: "swebench",
      tag: "latest",
      model_name_or_path: "test-api"
    }') \
  https://api.nejumi-swebench.org/v1/jobs | jq
```

レスポンス例：
```json
{
  "job_id": "job_1729612345678",
  "status": "queued",
  "created_at": 1729612345.678,
  "started_at": null,
  "finished_at": null
}
```

### 4. ジョブのステータス確認

```bash
JOB_ID=job_1729612345678
API_KEY=your_secure_api_key_here

# ステータス確認
curl -s -H "X-API-Key: $API_KEY" \
  https://api.nejumi-swebench.org/v1/jobs/$JOB_ID | jq

# ログ確認
curl -s -H "X-API-Key: $API_KEY" \
  https://api.nejumi-swebench.org/v1/jobs/$JOB_ID/logs
```

### 5. 評価の実行

設定ファイルを使って実際に評価を実行します：

```bash
cd /home/yuya/qwen3-next/llm-leaderboard

# 環境変数を設定
export SWE_API_KEY=your_secure_api_key_here
export OPENAI_API_KEY=your_openai_api_key
export WANDB_API_KEY=your_wandb_api_key

# 評価を実行
./myenv/bin/python scripts/run_eval.py \
  -c configs/config-gpt-4o-2024-11-20.yaml
```

---

## トラブルシューティング

### APIサーバーが起動しない

**問題**: `./myenv/bin/python が見つかりません`

**解決策**:
```bash
# 仮想環境を作成
python3 -m venv myenv

# 依存関係をインストール
./myenv/bin/pip install fastapi "uvicorn[standard]" swebench python-dotenv
```

---

### Docker権限エラー

**問題**: `permission denied while trying to connect to the Docker daemon socket`

**解決策**:
```bash
# ユーザーをdockerグループに追加
sudo usermod -aG docker $USER

# 新しいグループを適用（またはログアウト/ログイン）
newgrp docker

# 確認
docker ps
```

---

### APIキーエラー

**問題**: `401 Unauthorized`

**解決策**:
1. `.env` ファイルに `SWE_API_KEY` が設定されているか確認
2. リクエスト時に `X-API-Key` ヘッダーを正しく付けているか確認
3. サーバーを再起動して `.env` を再読み込み

```bash
# サーバーを停止
pkill -f swebench_server.py

# 再起動
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh
```

---

### Cloudflare Tunnelが接続できない

**問題**: トンネルが接続できない、またはドメインにアクセスできない

**解決策**:

1. **トンネルが起動しているか確認**:
   ```bash
   # systemdサービスの場合
   sudo systemctl status cloudflared
   
   # プロセスを確認
   ps aux | grep cloudflared
   ```

2. **DNS設定を確認**:
   ```bash
   # DNSが正しく設定されているか確認
   dig api.nejumi-swebench.org
   nslookup api.nejumi-swebench.org
   ```

3. **設定ファイルを確認**:
   ```bash
   cat ~/.cloudflared/config.yml
   ```

4. **トンネルを再起動**:
   ```bash
   sudo systemctl restart cloudflared
   
   # ログを確認
   sudo journalctl -u cloudflared -f
   ```

5. **ローカルでAPIサーバーが起動しているか確認**:
   ```bash
   curl http://localhost:8000/
   ```

---

### 評価がタイムアウトする

**問題**: ジョブが `timeout_sec` 以内に完了しない

**解決策**:
1. YAML設定でタイムアウト時間を延長:
   ```yaml
   swebench:
     api_server:
       timeout_sec: 3600  # 1時間に延長
   ```

2. サーバー側のワーカー数を増やす:
   ```bash
   # .env ファイルで設定
   echo "SWE_WORKERS=8" >> .env
   
   # サーバーを再起動
   pkill -f swebench_server.py
   ./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh
   ```

---

### Dockerイメージがプルできない

**問題**: Docker Hub rate limitに達した

**解決策**:
```bash
# Docker Hubにログイン
docker login

# またはconfigファイルを設定
# ~/.docker/config.json に認証情報を追加
```

---

## その他の有用なコマンド

### クリーンアップスクリプト

古い一時ファイルやログを削除：

```bash
cd /home/yuya/qwen3-next/llm-leaderboard
./scripts/evaluator/evaluate_utils/swebench_pkg/cleanup_temp.sh
```

### APIエンドポイント一覧

- `POST /v1/jobs` - ジョブの送信
- `GET /v1/jobs/{job_id}` - ジョブのステータス取得
- `GET /v1/jobs/{job_id}/logs` - ジョブのログ取得
- `GET /v1/summary` - サーバーサマリー情報

---

## まとめ

このマニュアルに従うことで：

1. ✅ SWE-Bench APIサーバーをローカルで起動
2. ✅ Cloudflare Tunnelで外部公開
3. ✅ YAML設定ファイルでAPIサーバーを利用
4. ✅ 評価を実行してスコアを取得

できるようになります。

質問や問題があれば、ログファイルを確認するか、GitHubのIssueで報告してください。

---

**作成日**: 2025-10-22
**バージョン**: 1.0

