# SWE-Bench APIサーバー ドキュメント一覧

このディレクトリには、SWE-Bench APIサーバーの立ち上げと運用に関するドキュメントがあります。

---

## 📚 ドキュメント一覧

### 🌟 推奨: 初めて立ち上げる場合

**[完全版立ち上げマニュアル](./swebench_server_startup_guide.md)**
- 実際の立ち上げ作業で発生した問題と解決方法を含む
- Python仮想環境のセットアップから動作確認まで
- よくある問題の詳細な解決方法
- systemdサービス化の手順
- Cloudflare Tunnelの設定
- **対象者**: 初めて設定する人、問題が発生して困っている人

---

### ⚡ 経験者向け: 素早く立ち上げたい場合

**[クイックスタートガイド](./swebench_api_quickstart.md)**
- 5分で起動する最小限の手順
- よく使うコマンドの一覧
- トラブルシューティング早見表
- **対象者**: 以前設定したことがある人、素早く再設定したい人

---

### 📖 詳細情報

**[完全セットアップガイド](./swebench_api_server_setup.md)**
- 前提条件の詳細な確認
- 依存関係のインストール手順
- Cloudflare Tunnelの詳細設定
- YAML設定ファイルの書き方
- 動作確認手順
- **対象者**: 詳細な説明が必要な人

**[サマリードキュメント](./swebench_api_summary_ja.md)**
- 全体のまとめと概要
- アーキテクチャ図
- セキュリティのポイント
- 実際の使用例
- **対象者**: 全体像を把握したい人

---

## 🎯 状況別おすすめドキュメント

### 🆕 初めてサーバーを立ち上げる
→ **[完全版立ち上げマニュアル](./swebench_server_startup_guide.md)**

理由: 実際に発生した問題と解決方法が含まれており、スムーズに立ち上げられます。

### ⚡ 以前立ち上げたことがある
→ **[クイックスタートガイド](./swebench_api_quickstart.md)**

理由: 必要最小限の手順でサクッと起動できます。

### 🐛 問題が発生して困っている
→ **[完全版立ち上げマニュアル > よくある問題と解決方法](./swebench_server_startup_guide.md#よくある問題と解決方法)**

理由: 実際に発生した問題とその解決方法が詳しく記載されています。

### 🌐 Cloudflare Tunnelで外部公開したい
→ **[完全セットアップガイド > Cloudflare Tunnelでの外部公開](./swebench_api_server_setup.md#cloudflare-tunnelでの外部公開)**

または

→ **[完全版立ち上げマニュアル > Cloudflare Tunnelでの外部公開](./swebench_server_startup_guide.md#cloudflare-tunnelでの外部公開)**

理由: Cloudflare Tunnelの設定手順が詳しく記載されています。

### 📊 全体像を把握したい
→ **[サマリードキュメント](./swebench_api_summary_ja.md)**

理由: アーキテクチャや使用例が図解されています。

---

## 🆘 よくある問題

### 問題1: `./myenv/bin/python が見つかりません`
**解決方法**: [完全版マニュアル - Python仮想環境が存在しない](./swebench_server_startup_guide.md#問題2-python仮想環境が存在しない)

### 問題2: `export: '#OpenRouter': not a valid identifier`
**解決方法**: [完全版マニュアル - .envファイルのコメント行エラー](./swebench_server_startup_guide.md#問題1-envファイルのコメント行エラー)

### 問題3: `permission denied /var/run/docker.sock`
**解決方法**: [完全版マニュアル - Docker権限エラー](./swebench_server_startup_guide.md#問題3-docker権限エラー)

### 問題4: `Address already in use`
**解決方法**: [完全版マニュアル - ポートが既に使用されている](./swebench_server_startup_guide.md#問題4-ポートが既に使用されている)

---

## 📝 基本的な立ち上げ手順（概要）

```bash
# 1. Python仮想環境を作成
python3 -m venv myenv

# 2. 依存関係をインストール
./myenv/bin/pip install fastapi "uvicorn[standard]" swebench python-dotenv

# 3. 環境変数を設定（.envファイル）
# SWE_API_KEY=your_key_here
# SWE_WORKERS=16

# 4. サーバーを起動
./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh

# 5. 動作確認
curl http://localhost:8000/v1/summary | jq
```

詳細は各ドキュメントを参照してください。

---

## 🔗 その他の関連ドキュメント

- [SWE-Bench評価詳細（日本語）](./README_swebench_ja.md)
- [SWE-Bench評価詳細（英語）](./README_swebench.md)
- [元のAPIサーバードキュメント](../scripts/evaluator/evaluate_utils/swebench_pkg/swebench_api_server.md)

---

**最終更新**: 2025-10-22

