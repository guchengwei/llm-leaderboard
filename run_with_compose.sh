#!/bin/bash
# run_with_compose.sh - .env + configs だけで Docker Compose 評価を実行
# 使い方: ./run_with_compose.sh <config-file-or-model> [-d]
set -euo pipefail

DEBUG=false
if [[ "$*" == *"-d"* ]]; then DEBUG=true; fi

[ $# -lt 1 ] && { echo "Usage: $0 <model|config> [-d]"; exit 1; }
TARGET=$1

# ----- 1. config ファイル確定 ------------------------------------------------
if [[ -f "configs/$TARGET" ]]; then CFG="configs/$TARGET"; else
  if [[ -f "configs/config-${TARGET}.yaml" ]]; then CFG="configs/config-${TARGET}.yaml"; else
    echo "Config not found: $TARGET"; exit 1;
  fi
fi
CFG_BASENAME=$(basename "$CFG")
export EVAL_CONFIG_PATH=$CFG_BASENAME

# api タイプを取得
API_TYPE=$(grep '^api:' "$CFG" | awk '{print $2}' | tr -d '"')
API_TYPE=${API_TYPE:-openai}

# VLLM_VERSIONをYAMLから取得（vllm_tagまたはvllm_docker_image）
VLLM_TAG=$(grep -E '^\s*vllm_tag:' "$CFG" | awk '{print $2}' | tr -d '"' || echo "")
VLLM_DOCKER_IMAGE=$(grep -E '^\s*vllm_docker_image:' "$CFG" | awk '{print $2}' | tr -d '"' || echo "")

# 優先順位: vllm_docker_image > vllm_tag > デフォルト(latest)
if [[ -n "$VLLM_DOCKER_IMAGE" ]]; then
    # フルイメージ名が指定されている場合はタグを抽出
    VLLM_VERSION=$(echo "$VLLM_DOCKER_IMAGE" | awk -F':' '{print $NF}')
    [[ "$VLLM_VERSION" == "$VLLM_DOCKER_IMAGE" ]] && VLLM_VERSION="latest"
elif [[ -n "$VLLM_TAG" ]]; then
    VLLM_VERSION="$VLLM_TAG"
else
    VLLM_VERSION="latest"
fi

export VLLM_VERSION

$DEBUG && echo "API_TYPE=$API_TYPE"
$DEBUG && echo "VLLM_VERSION=$VLLM_VERSION"

# Pre-create sandbox dependencies dir for dify-sandbox
mkdir -p ./volumes/sandbox/dependencies
if [ ! -f ./volumes/sandbox/dependencies/python-requirements.txt ]; then
  echo "INFO: Creating empty python-requirements.txt for dify-sandbox"
  touch ./volumes/sandbox/dependencies/python-requirements.txt
fi

# Pre-clean conflicting containers
for cname in llm-stack-vllm-1 llm-leaderboard; do
  if docker ps -a --format '{{.Names}}' | grep -q "^${cname}$"; then
    echo "🗑️  Removing stale container ${cname}"
    docker rm -f ${cname} 2>/dev/null || true
  fi
done

# ----- 2. 必要サービスを起動 --------------------------------------------------
# 共通: プロキシ & サンドボックス
SERVICES=(ssrf-proxy dify-sandbox)

# 依存関係のズレを避けるため、評価コンテナは毎回 build する。
# 特に openai-compatible 利用時は従来 llm-leaderboard が再 build されず、
# wandb / weave などの実行環境がリポジトリの lock とズレることがあった。
COMPOSE_ARGS=(--build)

if [[ "$API_TYPE" == vllm* ]]; then
  PROFILES=(--profile vllm-docker)
  SERVICES+=(vllm)
fi

$DEBUG && echo "docker compose ${PROFILES[*]} up -d ${COMPOSE_ARGS[*]} ${SERVICES[*]}"

docker compose ${PROFILES[@]:-} up -d ${COMPOSE_ARGS[@]:-} "${SERVICES[@]}"

# ----- 3. vLLM ヘルスチェック -------------------------------------------------
if [[ "$API_TYPE" == vllm* ]]; then
  echo -n "Waiting for vLLM ready"
  for i in {1..120}; do
    if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then echo " OK"; break; fi
    echo -n "."; sleep 5
    if [ $i -eq 120 ]; then echo "\nTimeout waiting vLLM"; exit 1; fi
  done
fi

# ----- 4. 評価実行 -----------------------------------------------------------
CMD="cd /workspace && source .venv/bin/activate && python -u scripts/run_eval.py --config $EVAL_CONFIG_PATH"

# llm-leaderboard は one-shot ジョブとして run する。
# up -d でサービス起動後に exec すると、サービス定義の command と exec の両方が走り、
# 評価が二重起動して W&B Run が重複する。
docker compose run --rm llm-leaderboard bash -c "$CMD"

# ----- 5. オプション: vLLM 停止 ------------------------------------------------
if [[ "$API_TYPE" == vllm* ]]; then
  read -p "Stop vLLM container? (y/N): " ans
  if [[ "${ans,,}" == y ]]; then docker compose stop vllm; fi
fi

echo "Done." 
