# SWE-Bench APIサーバー 立ち上げマニュアル（完全版）

このマニュアルは、実際の立ち上げ作業で発生した問題と解決方法を含む、実践的なガイドです。

---

## 📋 目次

1. [前提条件の確認](#前提条件の確認)
2. [ステップ1: Python仮想環境のセットアップ](#ステップ1-python仮想環境のセットアップ)
3. [ステップ2: 環境変数の設定](#ステップ2-環境変数の設定)
4. [ステップ3: サーバーの起動](#ステップ3-サーバーの起動)
5. [ステップ4: 動作確認](#ステップ4-動作確認)
6. [よくある問題と解決方法](#よくある問題と解決方法)
7. [サーバー管理](#サーバー管理)
8. [Cloudflare Tunnelでの外部公開](#cloudflare-tunnelでの外部公開)

---

## 前提条件の確認

以下がインストールされていることを確認してください：

### 1. Dockerのインストール確認

```bash
docker --version
```

**期待される出力例:**
```
Docker version 24.0.7, build afdd53b
```

Docker がインストールされていない場合：
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### 2. Docker権限の確認

```bash
docker ps
```

**エラーが出る場合（permission denied）:**
```bash
# ユーザーをdockerグループに追加
sudo usermod -aG docker $USER

# 新しいグループを適用（またはログアウト/ログイン）
newgrp docker

# 再度確認
docker ps
```

### 3. Python 3のインストール確認

```bash
python3 --version
```

**期待される出力例:**
```
Python 3.10.12
```

---

## ステップ1: Python仮想環境のセットアップ

### 1-1. リポジトリのルートに移動

```bash
cd /home/yuya/qwen3-next/llm-leaderboard
```

### 1-2. 仮想環境の作成

```bash
# 仮想環境を作成
python3 -m venv myenv
```

**確認:**
```bash
ls -la myenv/
```

以下のディレクトリが存在すれば成功：
- `bin/`
- `lib/`
- `include/`

### 1-3. pipのアップグレード

```bash
./myenv/bin/pip install --upgrade pip
```

### 1-4. 必要な依存関係のインストール

```bash
./myenv/bin/pip install fastapi "uvicorn[standard]" swebench python-dotenv
```

**インストールされるパッケージ（主要なもの）:**
- `fastapi` - APIサーバーフレームワーク
- `uvicorn` - ASGIサーバー
- `swebench` - SWE-Benchツール
- `python-dotenv` - 環境変数管理
- その他の依存関係（自動的にインストールされる）

**確認:**
```bash
./myenv/bin/python -c "import fastapi; import uvicorn; import swebench; print('All packages installed successfully!')"
```

---

## ステップ2: 環境変数の設定

### 2-1. `.env`ファイルの確認

リポジトリのルートに`.env`ファイルが存在するか確認：

```bash
ls -la .env
```

### 2-2. `.env`ファイルの内容確認

```bash
cat .env | grep -E "^SWE_|^#"
```

### 2-3. 必要な環境変数の追加

`.env`ファイルに以下が設定されているか確認（なければ追加）：

```bash
# SWE-Bench APIサーバー設定
SWE_API_KEY=nejumi0123456789
SWE_WORKERS=16
PORT=8000
```

**注意: 行末コメントの問題**

❌ **間違った書き方（エラーになる）:**
```bash
OPENAI_COMPATIBLE_API_KEY=sk-or-v1-xxxxx #OpenRouter
```

✅ **正しい書き方:**
```bash
# OpenRouter
OPENAI_COMPATIBLE_API_KEY=sk-or-v1-xxxxx
```

または

```bash
OPENAI_COMPATIBLE_API_KEY=sk-or-v1-xxxxx
# OpenRouter
```

**理由:** 起動スクリプト`start_server.sh`は行末コメントを処理できるように修正済みですが、念のため上記の書き方を推奨します。

### 2-4. 環境変数の値の説明

| 環境変数 | 説明 | デフォルト値 |
|---------|------|-------------|
| `SWE_API_KEY` | APIアクセス用のキー。設定すると全リクエストで`X-API-Key`ヘッダーが必須になる | なし |
| `SWE_WORKERS` | 並列処理のワーカー数。CPUコア数に応じて調整 | 4 |
| `PORT` | サーバーのポート番号 | 8000 |

**推奨ワーカー数:**
- 4コア: 3-4ワーカー
- 8コア: 6-8ワーカー
- 16コア: 12-16ワーカー
- 32コア以上: 24ワーカー

---

## ステップ3: サーバーの起動

### 3-1. 起動スクリプトの実行

```bash
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh
```

**期待される出力:**
```
Server started on http://0.0.0.0:8000
Logs: /tmp/swebench_server.out
```

### 3-2. 起動確認

2秒待ってからログを確認：

```bash
sleep 2
tail -20 /tmp/swebench_server.out
```

**正常起動時の出力例:**
```
INFO:     Started server process [1136179]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 3-3. カスタムホスト・ポートで起動

デフォルト以外のホスト・ポートで起動する場合：

```bash
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh --host 127.0.0.1 --port 8080
```

---

## ステップ4: 動作確認

### 4-1. サーバーステータスの確認

```bash
curl -s http://localhost:8000/v1/summary | jq
```

**期待される出力:**
```json
{
  "status": "ok",
  "workers": 16,
  "jobs": {
    "queued": 0,
    "running": 0,
    "finished": 0,
    "failed": 0
  },
  "time": "2025-10-22T11:35:52Z"
}
```

### 4-2. プロセスの確認

```bash
ps aux | grep swebench_server
```

**期待される出力:**
```
yuya     1136179  0.0  0.1  12345 67890 ?        S    11:35   0:00 ./myenv/bin/python scripts/server/swebench_server.py --host 0.0.0.0 --port 8000
```

### 4-3. ポートの確認

```bash
sudo netstat -tlnp | grep :8000
```

または

```bash
sudo ss -tlnp | grep :8000
```

**期待される出力:**
```
tcp        0      0 0.0.0.0:8000            0.0.0.0:*               LISTEN      1136179/python
```

---

## よくある問題と解決方法

### 問題1: `.env`ファイルのコメント行エラー

**エラーメッセージ:**
```
export: `#OpenRouter': not a valid identifier
```

**原因:**
`.env`ファイルに行末コメントが含まれている
```bash
OPENAI_COMPATIBLE_API_KEY=sk-or-v1-xxxxx #OpenRouter
```

**解決方法:**

#### 方法A: `.env`ファイルを修正（推奨）

```bash
# 行末コメントを別の行に移動
sed -i 's/\(.*\) #\(.*\)$/\1\n# \2/' .env
```

または手動で編集：
```bash
nano .env
```

行末コメントを削除または別の行に移動

#### 方法B: 起動スクリプトを修正（既に修正済み）

`start_server.sh`スクリプトは行末コメントを処理できるように修正済みです。

---

### 問題2: Python仮想環境が存在しない

**エラーメッセージ:**
```
[ERROR] ./myenv/bin/python が見つかりません。仮想環境を作成・依存をインストールしてください。
```

**原因:**
`myenv`ディレクトリが存在しないか、Python実行ファイルがない

**解決方法:**

```bash
# 仮想環境を作成
python3 -m venv myenv

# pipをアップグレード
./myenv/bin/pip install --upgrade pip

# 依存関係をインストール
./myenv/bin/pip install fastapi "uvicorn[standard]" swebench python-dotenv
```

---

### 問題3: Docker権限エラー

**エラーメッセージ:**
```
permission denied while trying to connect to the Docker daemon socket
```

**原因:**
現在のユーザーがDockerグループに所属していない

**解決方法:**

```bash
# ユーザーをdockerグループに追加
sudo usermod -aG docker $USER

# 新しいグループを適用
newgrp docker

# サーバーを再起動
pkill -f swebench_server.py
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh
```

---

### 問題4: ポートが既に使用されている

**エラーメッセージ:**
```
OSError: [Errno 98] Address already in use
```

**原因:**
ポート8000が既に使用されている

**解決方法:**

#### 方法A: 既存のプロセスを停止

```bash
# ポートを使用しているプロセスを確認
sudo lsof -i :8000

# プロセスを停止
pkill -f swebench_server.py

# 再度起動
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh
```

#### 方法B: 別のポートを使用

```bash
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh --port 8080
```

---

### 問題5: 依存関係のインストールエラー

**エラーメッセージ:**
```
ERROR: Could not find a version that satisfies the requirement ...
```

**原因:**
- インターネット接続の問題
- Pythonバージョンの非互換性

**解決方法:**

```bash
# Pythonバージョンを確認（3.8以上が必要）
python3 --version

# pipをアップグレード
./myenv/bin/pip install --upgrade pip

# 依存関係を再インストール
./myenv/bin/pip install --no-cache-dir fastapi "uvicorn[standard]" swebench python-dotenv
```

---

## サーバー管理

### サーバーの停止

```bash
pkill -f swebench_server.py
```

**確認:**
```bash
ps aux | grep swebench_server
```

### サーバーの再起動

```bash
# 停止
pkill -f swebench_server.py

# 起動
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh
```

### ログの確認

#### リアルタイムでログを確認

```bash
tail -f /tmp/swebench_server.out
```

#### 最新100行を確認

```bash
tail -100 /tmp/swebench_server.out
```

#### エラーのみを確認

```bash
grep -i error /tmp/swebench_server.out
```

### サーバーステータスの確認

```bash
curl -s http://localhost:8000/v1/summary | jq
```

### 自動起動の設定（systemdサービス）

恒久的にサーバーを稼働させる場合：

#### 1. サービスファイルを作成

```bash
sudo tee /etc/systemd/system/swebench-api.service << 'EOF'
[Unit]
Description=SWE-Bench API Server
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=yuya
WorkingDirectory=/home/yuya/qwen3-next/llm-leaderboard
EnvironmentFile=/home/yuya/qwen3-next/llm-leaderboard/.env
ExecStart=/home/yuya/qwen3-next/llm-leaderboard/myenv/bin/python /home/yuya/qwen3-next/llm-leaderboard/scripts/server/swebench_server.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

#### 2. サービスを有効化・起動

```bash
# サービスをリロード
sudo systemctl daemon-reload

# サービスを有効化（自動起動）
sudo systemctl enable swebench-api

# サービスを起動
sudo systemctl start swebench-api

# ステータス確認
sudo systemctl status swebench-api
```

#### 3. サービスの管理

```bash
# 起動
sudo systemctl start swebench-api

# 停止
sudo systemctl stop swebench-api

# 再起動
sudo systemctl restart swebench-api

# ステータス確認
sudo systemctl status swebench-api

# ログ確認
sudo journalctl -u swebench-api -f
```

---

## Cloudflare Tunnelでの外部公開

サーバーを外部からアクセス可能にする場合は、Cloudflare Tunnelを使用します。

### 1. cloudflaredのインストール

```bash
# 最新版をダウンロード
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64

# 実行権限を付与
chmod +x cloudflared-linux-amd64

# システムにインストール
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# バージョン確認
cloudflared --version
```

### 2. Cloudflareアカウントでログイン

```bash
cloudflared tunnel login
```

ブラウザが開くので、Cloudflareアカウントでログインし、使用するドメインを選択します。

### 3. トンネルの作成

```bash
# トンネルを作成
cloudflared tunnel create swebench-api
```

**出力例:**
```
Created tunnel swebench-api with id: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

**重要:** トンネルIDをメモしておいてください。

### 4. DNS設定

```bash
# サブドメインをトンネルにルーティング
cloudflared tunnel route dns swebench-api api.nejumi-swebench.org
```

**注意:** `api.nejumi-swebench.org`を自分のドメインに変更してください。

### 5. 設定ファイルの作成

```bash
# トンネルIDを環境変数に設定（上記で取得したID）
TUNNEL_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# 設定ファイルを作成
cat > ~/.cloudflared/config.yml << EOF
tunnel: ${TUNNEL_ID}
credentials-file: /home/yuya/.cloudflared/${TUNNEL_ID}.json

ingress:
  - hostname: api.nejumi-swebench.org
    service: http://localhost:8000
  - service: http_status:404
EOF
```

### 6. トンネルの起動

#### 方法A: 手動起動（テスト用）

```bash
# フォアグラウンドで起動
cloudflared tunnel run swebench-api

# または、バックグラウンドで起動
nohup cloudflared tunnel run swebench-api >/tmp/cloudflared.log 2>&1 & disown

# ログ確認
tail -f /tmp/cloudflared.log
```

#### 方法B: systemdサービスとして起動（推奨）

```bash
# サービスファイルをインストール
sudo cloudflared service install

# サービスを有効化
sudo systemctl enable cloudflared

# サービスを起動
sudo systemctl start cloudflared

# ステータス確認
sudo systemctl status cloudflared

# ログ確認
sudo journalctl -u cloudflared -f
```

### 7. 動作確認

```bash
# 外部からアクセス可能か確認
curl https://api.nejumi-swebench.org/v1/summary | jq
```

---

## YAML設定ファイルの設定

評価時にAPIサーバーを使用するには、設定YAMLファイルで以下のように指定します。

### ローカル環境の場合

```yaml
swebench:
  background_eval: true
  api_server:
    enabled: true
    endpoint: 'http://127.0.0.1:8000'
    api_key: null  # 環境変数 SWE_API_KEY を使用
    timeout_sec: 1200
```

### Cloudflare Tunnel経由の場合

```yaml
swebench:
  background_eval: true
  api_server:
    enabled: true
    endpoint: 'https://api.nejumi-swebench.org'
    api_key: null  # 環境変数 SWE_API_KEY を使用
    timeout_sec: 1200
```

---

## チェックリスト

### 初回セットアップ時

- [ ] Dockerがインストール済み（`docker --version`）
- [ ] Docker権限がある（`docker ps`が実行できる）
- [ ] Python 3.8以上がインストール済み（`python3 --version`）
- [ ] Python仮想環境`myenv`を作成済み
- [ ] 依存関係をインストール済み（`fastapi`, `uvicorn`, `swebench`）
- [ ] `.env`ファイルに`SWE_API_KEY`を設定済み
- [ ] `.env`ファイルに行末コメントがないことを確認
- [ ] サーバーが起動できることを確認

### 外部公開する場合（追加）

- [ ] Cloudflareアカウントがある
- [ ] 使用するドメインがある
- [ ] `cloudflared`がインストール済み
- [ ] トンネルを作成済み
- [ ] DNS設定が完了している
- [ ] 設定ファイル`~/.cloudflared/config.yml`を作成済み
- [ ] トンネルが起動していることを確認

---

## トラブルシューティングコマンド集

```bash
# サーバーが起動しているか確認
ps aux | grep swebench_server

# ポートが使用されているか確認
sudo netstat -tlnp | grep :8000

# ログの確認
tail -100 /tmp/swebench_server.out

# エラーログの確認
grep -i error /tmp/swebench_server.out

# サーバーの停止
pkill -f swebench_server.py

# Docker権限の確認
docker ps

# Python仮想環境の確認
ls -la myenv/bin/python

# 依存関係の確認
./myenv/bin/pip list | grep -E "fastapi|uvicorn|swebench"

# 環境変数の確認
cat .env | grep SWE_

# Cloudflare Tunnelのステータス確認
sudo systemctl status cloudflared

# Cloudflare Tunnelのログ確認
sudo journalctl -u cloudflared -n 100
```

---

## まとめ

このマニュアルに従うことで：

✅ Python仮想環境を正しくセットアップ
✅ 環境変数を適切に設定
✅ SWE-Bench APIサーバーを起動
✅ サーバーの動作を確認
✅ 一般的な問題を解決
✅ （オプション）Cloudflare Tunnelで外部公開

できるようになります。

---

**作成日**: 2025-10-22
**最終更新**: 2025-10-22
**バージョン**: 2.0（実際の立ち上げ作業の経験を反映）

