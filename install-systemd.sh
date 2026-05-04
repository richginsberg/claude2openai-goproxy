#!/usr/bin/env bash
set -euo pipefail

# Defaults (override with environment variables or arguments)
BIN_DIR="${BIN_DIR:-/usr/local/bin}"
BIN_NAME="golangproxy"
VLLM_URL="${VLLM_URL:-http://localhost:6006}"
VLLM_MODEL="${VLLM_MODEL:-Lorbus/Qwen3.6-27B-int4-AutoRound}"
PROXY_PORT="${PROXY_PORT:-4000}"
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_TEMPLATE="${SCRIPT_DIR}/anthropic-proxy.service"
SERVICE_DIR="${HOME}/.config/systemd/user"

echo "=== golangproxy systemd install ==="
echo "  Binary:      ${BIN_DIR}/${BIN_NAME}"
echo "  vLLM URL:    ${VLLM_URL}"
echo "  vLLM Model:  ${VLLM_MODEL}"
echo "  Port:        ${PROXY_PORT}"
echo "  Host:        ${PROXY_HOST}"

# Ensure binary is in place
if [ ! -f "${BIN_DIR}/${BIN_NAME}" ]; then
    echo ""
    echo "Binary not found at ${BIN_DIR}/${BIN_NAME}. Installing..."
    if [ ! -f "${SCRIPT_DIR}/${BIN_NAME}" ]; then
        echo "  Building binary..."
        (cd "${SCRIPT_DIR}" && go build -o "${BIN_NAME}" main.go)
    fi
    sudo cp "${SCRIPT_DIR}/${BIN_NAME}" "${BIN_DIR}/${BIN_NAME}"
    echo "  Installed to ${BIN_DIR}/${BIN_NAME}"
else
    echo ""
    echo "Binary already exists at ${BIN_DIR}/${BIN_NAME}"
fi

# Generate service file from template
mkdir -p "${SERVICE_DIR}"
sed \
    -e "s|__GOLANGPROXY_BIN__|${BIN_DIR}/${BIN_NAME}|" \
    -e "s|__VLLM_URL__|${VLLM_URL}|" \
    -e "s|__VLLM_MODEL__|${VLLM_MODEL}|" \
    -e "s|__PROXY_PORT__|${PROXY_PORT}|" \
    -e "s|__PROXY_HOST__|${PROXY_HOST}|" \
    "${SERVICE_TEMPLATE}" > "${SERVICE_DIR}/anthropic-proxy.service"

echo "  Service file written to ${SERVICE_DIR}/anthropic-proxy.service"

# Enable and start
systemctl --user daemon-reload
systemctl --user enable anthropic-proxy.service
systemctl --user restart anthropic-proxy.service

echo ""
echo "Done. Check status with:"
echo "  systemctl --user status anthropic-proxy.service"
echo "  journalctl --user -u anthropic-proxy.service -f"
