#!/bin/bash
# Post-create script for devcontainer setup
# This runs once after the container is created

set -e

# Source .env file if it exists
if [ -f /workspace/.env ]; then
    set -a
    source /workspace/.env
    set +a
fi

echo "Running post-create setup..."

# Set up SSH keys from host
mkdir -p /home/node/.ssh
cp -r /home/node/.ssh-host/* /home/node/.ssh/ 2>/dev/null || true
chmod 700 /home/node/.ssh
chmod 600 /home/node/.ssh/id_* 2>/dev/null || true
chmod 644 /home/node/.ssh/*.pub 2>/dev/null || true
git remote set-url origin git@github.com:quantitative-mri-and-in-vivo-histology/microct_io_vis.git

# Install Claude Code if enabled
if [ "${ENABLE_CLAUDE:-0}" = "1" ]; then
    echo "Installing Claude Code..."
    npm install -g @anthropic-ai/claude-code 2>/dev/null || true
    echo "Installing Claude Code VS Code extension..."
    code --install-extension anthropic.claude-code 2>/dev/null || true
fi

echo "Post-create setup complete!"
