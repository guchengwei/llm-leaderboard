# SWE-Bench APIサーバー クイックスタートガイド

5分でSWE-Bench APIサーバーを起動する最小限の手順です。

詳細は [完全版マニュアル](./swebench_server_startup_guide.md) を参照してください。

---

## 🚀 クイックスタート

### 1. セットアップ

```bash
cd /home/yuya/qwen3-next/llm-leaderboard

# Python仮想環境を作成（初回のみ）
python3 -m venv myenv
./myenv/bin/pip install --upgrade pip
./myenv/bin/pip install fastapi "uvicorn[standard]" swebench python-dotenv

# 環境変数を設定（.envに以下を追加）
# SWE_API_KEY=your_key_here
# SWE_WORKERS=16
```

### 2. サーバー起動

```bash
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh
```

### 3. 動作確認

```bash
curl http://localhost:8000/v1/summary | jq
```

---

## 🌐 Cloudflare Tunnel（外部公開）

```bash
# インストール
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# トンネル作成
cloudflared tunnel login
cloudflared tunnel create swebench-api
cloudflared tunnel route dns swebench-api api.your-domain.com

# 設定ファイル（~/.cloudflared/config.yml）
# tunnel: YOUR_TUNNEL_ID
# credentials-file: /home/yuya/.cloudflared/YOUR_TUNNEL_ID.json
# ingress:
#   - hostname: api.your-domain.com
#     service: http://localhost:8000
#   - service: http_status:404

# 起動（systemdサービス推奨）
sudo cloudflared service install
sudo systemctl enable --now cloudflared

# または手動起動
TOKEN=$(cloudflared tunnel token swebench-api 2>/dev/null | head -1)
nohup cloudflared tunnel --config ~/.cloudflared/config.yml run --token $TOKEN >/tmp/cloudflared.log 2>&1 & disown
```

---

## 📝 YAML設定

```yaml
swebench:
  background_eval: true
  api_server:
    enabled: true
    endpoint: 'https://api.your-domain.com'  # または http://localhost:8000
    api_key: null  # 環境変数 SWE_API_KEY を使用
    timeout_sec: 1200
```

---

## 🔧 管理コマンド

```bash
# サーバー管理
pkill -f swebench_server.py                              # 停止
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh  # 起動
tail -f /tmp/swebench_server.out                         # ログ

# トンネル管理
sudo systemctl {start|stop|status} cloudflared           # systemd
pkill cloudflared                                         # 手動停止

# APIテスト
curl http://localhost:8000/v1/summary | jq              # ステータス
curl -H "X-API-Key: $SWE_API_KEY" -H "Content-Type: application/json" \
  -d '{"instance_id":"astropy__astropy-12907","patch_diff":"...","namespace":"swebench","tag":"latest"}' \
  http://localhost:8000/v1/jobs | jq                     # ジョブ送信
```

---

## 🐛 トラブルシューティング

| エラー | 解決方法 |
|--------|---------|
| `./myenv/bin/python が見つかりません` | `python3 -m venv myenv` |
| `permission denied /var/run/docker.sock` | `sudo usermod -aG docker $USER && newgrp docker` |
| `Address already in use` | `pkill -f swebench_server.py` |

詳細: [完全版マニュアル](./swebench_server_startup_guide.md#よくある問題と解決方法)

---

## 📚 関連ドキュメント

- [完全版立ち上げマニュアル](./swebench_server_startup_guide.md) - 詳細な手順とトラブルシューティング
- [完全セットアップガイド](./swebench_api_server_setup.md) - Cloudflare Tunnel詳細
- [SWE-Bench評価詳細](./README_swebench_ja.md)

---

**最終更新**: 2025-10-22
