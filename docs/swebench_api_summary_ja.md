# SWE-Bench APIサーバー設定ガイド まとめ

このドキュメントは、SWE-Bench APIサーバーの立ち上げとCloudflare Tunnelの設定に関する作成したドキュメントのまとめです。

---

## 📚 作成したドキュメント

### 1. [完全版立ち上げマニュアル](./swebench_server_startup_guide.md) ⭐ **推奨**
   - **用途**: 初めて立ち上げる場合、または問題が発生した場合
   - **内容**:
     - 実際の立ち上げ作業で発生した問題と解決方法
     - Python仮想環境のセットアップ（詳細手順）
     - 環境変数の設定（注意点含む）
     - サーバーの起動と確認
     - よくある問題と解決方法（実例ベース）
     - サーバー管理（systemdサービス化含む）
     - Cloudflare Tunnelの設定
     - トラブルシューティングコマンド集
   - **対象者**: 初めて設定する人、問題が発生して困っている人
   - **バージョン**: 2.0（実際の経験を反映）

### 2. [クイックスタートガイド](./swebench_api_quickstart.md)
   - **用途**: 素早く立ち上げたい場合
   - **内容**:
     - 5分で起動する最小限の手順
     - よくある問題の早見表
     - よく使うコマンドの一覧
     - トラブルシューティング早見表
     - チェックリスト
   - **対象者**: 以前設定したことがある人、素早く再設定したい人

### 3. [完全セットアップガイド](./swebench_api_server_setup.md)
   - **用途**: 詳細な手順書が必要な場合
   - **内容**:
     - 前提条件の確認
     - 依存関係のインストール手順
     - 環境変数の設定方法
     - APIサーバーの起動方法（2種類）
     - Cloudflare Tunnelの詳細設定
     - YAML設定ファイルの書き方
     - 動作確認手順
     - 詳細なトラブルシューティング
   - **対象者**: 詳細な説明が必要な人

---

## 🚀 立ち上げ方の概要

### ステップ1: 依存関係のインストール

```bash
cd /home/yuya/qwen3-next/llm-leaderboard
./myenv/bin/pip install fastapi "uvicorn[standard]" swebench python-dotenv
```

### ステップ2: 環境変数の設定

`.env` ファイルに以下を追加：

```bash
SWE_API_KEY=your_secure_api_key_here
SWE_WORKERS=4
PORT=8000
```

### ステップ3: サーバー起動

```bash
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh
```

### ステップ4（オプション）: Cloudflare Tunnelで外部公開

```bash
# 1. cloudflaredのインストール
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# 2. ログインとトンネル作成
cloudflared tunnel login
cloudflared tunnel create swebench-api

# 3. DNS設定
cloudflared tunnel route dns swebench-api api.nejumi-swebench.org

# 4. 設定ファイル作成（~/.cloudflared/config.yml）
# 詳細は完全セットアップガイド参照

# 5. トンネル起動
sudo cloudflared service install
sudo systemctl start cloudflared
```

---

## ⚙️ YAML設定ファイルの書き方

評価実行時に使用する設定ファイル（例: `configs/config-gpt-4o-2024-11-20.yaml`）：

```yaml
swebench:
  background_eval: true
  api_server:
    enabled: true
    endpoint: 'https://api.nejumi-swebench.org'  # Cloudflare Tunnelのドメイン
    # または 'http://127.0.0.1:8000' でローカル実行
    api_key: null  # 環境変数 SWE_API_KEY が使用される
    timeout_sec: 1200
```

---

## 🔍 よく使うコマンド

### サーバー管理

```bash
# 起動
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh

# 停止
pkill -f swebench_server.py

# ログ確認
tail -f /tmp/swebench_server.out

# ステータス確認
curl http://localhost:8000/v1/summary | jq
```

### Cloudflare Tunnel管理

```bash
# 起動（systemdサービス）
sudo systemctl start cloudflared

# 停止
sudo systemctl stop cloudflared

# ステータス確認
sudo systemctl status cloudflared

# ログ確認
sudo journalctl -u cloudflared -f
```

### 動作確認

```bash
# ローカルでの確認
curl http://localhost:8000/v1/summary | jq

# 外部からの確認（Cloudflare Tunnel経由）
curl https://api.nejumi-swebench.org/v1/summary | jq

# ジョブの送信テスト
curl -s -H "Content-Type: application/json" \
  -H "X-API-Key: $SWE_API_KEY" \
  -d '{
    "instance_id": "astropy__astropy-12907",
    "patch_diff": "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new\n",
    "namespace": "swebench",
    "tag": "latest"
  }' \
  http://localhost:8000/v1/jobs | jq
```

---

## 📊 アーキテクチャ

```
┌─────────────────┐
│  評価スクリプト │  (run_eval.py with YAML config)
└────────┬────────┘
         │ HTTP Request (with X-API-Key)
         ▼
┌─────────────────┐
│ Cloudflare      │  (Optional, for external access)
│ Tunnel          │  api.nejumi-swebench.org
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SWE-Bench API   │  localhost:8000
│ Server          │  (FastAPI + Uvicorn)
│ (Python)        │
└────────┬────────┘
         │ Docker API
         ▼
┌─────────────────┐
│ Docker Daemon   │  SWE-Bench evaluation containers
│                 │  (swebench/sweb.eval.x86_64.*)
└─────────────────┘
```

---

## 🔐 セキュリティのポイント

1. **APIキーの設定**:
   - `.env` ファイルに `SWE_API_KEY` を設定
   - 全てのAPIリクエストで `X-API-Key` ヘッダーが必須
   - APIキーは秘密にして、公開リポジトリにコミットしない

2. **Cloudflare Tunnelのメリット**:
   - ファイアウォールの設定不要
   - 固定IPアドレス不要
   - SSL/TLS証明書は自動管理
   - DDoS保護が含まれる
   - ポートフォワーディング不要

3. **Docker権限**:
   - サーバーはDockerソケットへのアクセスが必要
   - 適切なユーザー権限を設定（`docker` グループに追加）

---

## 🐛 トラブルシューティング早見表

### 立ち上げ時によく発生する問題

| 症状 | エラーメッセージ | 原因 | 解決方法 |
|------|----------------|------|----------|
| **Python仮想環境がない** | `./myenv/bin/python が見つかりません` | 仮想環境が未作成 | `python3 -m venv myenv` で作成 |
| **行末コメントエラー** | `export: '#OpenRouter': not a valid identifier` | `.env`ファイルに行末コメント | `.env`から`#`以降を削除または別行に移動 |
| **Docker権限エラー** | `permission denied /var/run/docker.sock` | Docker権限がない | `sudo usermod -aG docker $USER && newgrp docker` |
| **依存関係エラー** | `Could not find a version...` | pip/Python環境の問題 | `./myenv/bin/pip install --upgrade pip` |
| **ポート使用中** | `Address already in use` | 既存プロセスが稼働中 | `pkill -f swebench_server.py` |

### 実行時の問題

| 症状 | 原因 | 解決方法 |
|------|------|----------|
| `401 Unauthorized` | APIキーが一致しない | `.env`の`SWE_API_KEY`を確認、`X-API-Key`ヘッダーを確認 |
| サーバーが起動しない | Python環境の問題 | `./myenv/bin/python`の存在確認、依存関係の再インストール |
| トンネルが繋がらない | cloudflaredの問題 | `sudo systemctl status cloudflared`でステータス確認 |
| タイムアウトする | 評価時間が長い | YAML設定で`timeout_sec`を延長（例: 3600） |
| Dockerイメージがプルできない | rate limit | Docker Hubにログイン: `docker login` |

**詳細な解決方法は[完全版マニュアル](./swebench_server_startup_guide.md#よくある問題と解決方法)を参照してください。**

---

## 📖 関連ドキュメント

- **[完全版立ち上げマニュアル](./swebench_server_startup_guide.md)** - 実際の問題と解決方法を含む詳細な手順書 ⭐ **推奨**
- [クイックスタートガイド](./swebench_api_quickstart.md) - 5分で起動
- [完全セットアップガイド](./swebench_api_server_setup.md) - Cloudflare Tunnel含む詳細ガイド
- [SWE-Bench評価詳細（日本語）](./README_swebench_ja.md) - 評価方法の詳細
- [SWE-Bench評価詳細（英語）](./README_swebench.md) - 評価方法の詳細（英語版）
- [BFCL評価詳細](./README_bfcl_ja.md) - BFCL評価の詳細

---

## 💡 Tips

### ローカルテストと本番環境の切り替え

YAML設定ファイルで `endpoint` を変更するだけで切り替え可能：

```yaml
# ローカルテスト
swebench:
  api_server:
    enabled: true
    endpoint: 'http://127.0.0.1:8000'

# 本番環境（Cloudflare Tunnel経由）
swebench:
  api_server:
    enabled: true
    endpoint: 'https://api.nejumi-swebench.org'
```

### 複数のワーカーで並列処理

`.env` ファイルで `SWE_WORKERS` を調整：

```bash
# CPUコア数に応じて調整（推奨: CPUコア数 * 0.75）
SWE_WORKERS=8
```

### ログのローテーション

長期運用する場合は、ログファイルのローテーションを設定：

```bash
# logrotateの設定例
sudo tee /etc/logrotate.d/swebench-api << EOF
/tmp/swebench_server.out {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOF
```

---

## 🎯 使用例

### 1. ローカルで評価を実行

```bash
cd /home/yuya/qwen3-next/llm-leaderboard

# サーバー起動
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh

# 評価実行（ローカルエンドポイント使用）
export SWE_API_KEY=your_key_here
./myenv/bin/python scripts/run_eval.py -c configs/config-gpt-4o-2024-11-20.yaml
```

### 2. Cloudflare Tunnel経由で外部から評価

```bash
# サーバーとトンネルを起動
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh
sudo systemctl start cloudflared

# 評価実行（外部エンドポイント使用）
# YAML設定で endpoint: 'https://api.nejumi-swebench.org' を指定
./myenv/bin/python scripts/run_eval.py -c configs/config-gpt-4o-2024-11-20.yaml
```

---

## ✅ チェックリスト

### 初回セットアップ時

- [ ] Dockerがインストール済み（`docker --version`）
- [ ] Docker権限がある（`docker ps`が実行できる）
- [ ] Python仮想環境 `./myenv` が存在する
- [ ] 依存関係インストール済み（`fastapi`, `uvicorn`, `swebench`）
- [ ] `.env` ファイルに `SWE_API_KEY` を設定
- [ ] サーバーが起動できる（`curl http://localhost:8000/v1/summary`）
- [ ] （外部公開する場合）Cloudflareアカウントがある
- [ ] （外部公開する場合）Cloudflare Tunnelが設定済み
- [ ] （外部公開する場合）DNS設定が完了している
- [ ] YAML設定ファイルが正しく設定されている

### 運用時

- [ ] サーバーが起動している（`ps aux | grep swebench_server`）
- [ ] （外部公開の場合）Cloudflare Tunnelが起動している（`sudo systemctl status cloudflared`）
- [ ] ログファイルを定期的に確認（`tail -f /tmp/swebench_server.out`）
- [ ] ディスク容量に余裕がある（一時ファイルが蓄積する）
- [ ] Docker Hubの認証情報が有効（rate limit対策）

---

**作成日**: 2025-10-22
**最終更新**: 2025-10-22

