#!/bin/bash
# Post-start script - runs every time the container starts

# Source .env file if it exists
if [ -f /workspace/.env ]; then
    set -a
    source /workspace/.env
    set +a
fi

# Install Claude Code VS Code extension if enabled
if [ "${ENABLE_CLAUDE:-0}" = "1" ]; then
    echo "Installing Claude Code VS Code extension..."
    code --install-extension anthropic.claude-code 2>&1 || true
fi
