FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ARG TZ
ENV TZ="$TZ"
ENV DEBIAN_FRONTEND=noninteractive

ARG CLAUDE_CODE_VERSION=latest

# Install Python 3.11 from deadsnakes PPA (cuda base image ships no python)
RUN apt-get update && \
  apt-get install -y --no-install-recommends software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x (for optional Claude Code install via post-create)
RUN apt-get update && \
  apt-get install -y --no-install-recommends curl ca-certificates gnupg && \
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
  apt-get install -y nodejs && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# Install development tools (matches CPU variant)
RUN apt-get update && apt-get install -y --no-install-recommends \
  less \
  git \
  openssh-client \
  procps \
  sudo \
  fzf \
  zsh \
  man-db \
  unzip \
  gnupg2 \
  gh \
  iptables \
  ipset \
  iproute2 \
  dnsutils \
  aggregate \
  jq \
  nano \
  vim \
  wget \
  build-essential \
  gfortran \
  libopenblas-dev \
  liblapack-dev \
  pkg-config \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04's fzf deb doesn't ship the zsh shell snippets that oh-my-zsh's
# fzf plugin expects at /usr/share/doc/fzf/examples/. Fetch them from upstream
# (pinned to the fzf version in jammy) so zsh-in-docker's fzf plugin works.
RUN mkdir -p /usr/share/doc/fzf/examples && \
  wget -qO /usr/share/doc/fzf/examples/key-bindings.zsh \
    https://raw.githubusercontent.com/junegunn/fzf/0.29.0/shell/key-bindings.zsh && \
  wget -qO /usr/share/doc/fzf/examples/completion.zsh \
    https://raw.githubusercontent.com/junegunn/fzf/0.29.0/shell/completion.zsh

# Fix pip for Python 3.11
RUN python3 -m pip install --upgrade pip setuptools

# Create node user (UID 1000 to match typical dev environments)
ARG USERNAME=node
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
  useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME && \
  echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
  chmod 0440 /etc/sudoers.d/$USERNAME

# Ensure node user has access to /usr/local/share
RUN mkdir -p /usr/local/share/npm-global && \
  chown -R node:node /usr/local/share

# Persist bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME /commandhistory

ENV DEVCONTAINER=true

# Create workspace directory and set permissions
RUN mkdir -p /workspace && \
  chown -R node:node /workspace

WORKDIR /workspace

# Install git-delta
ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
  wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

USER node

# Install global packages
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin

ENV SHELL=/bin/zsh
ENV EDITOR=nano
ENV VISUAL=nano

# Default powerline10k theme
ARG ZSH_IN_DOCKER_VERSION=1.2.0
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
  -p git \
  -p fzf \
  -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
  -a "source /usr/share/doc/fzf/examples/completion.zsh" \
  -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  -x

# Install Claude Code globally (GPU variant bakes it in rather than relying on
# the ENABLE_CLAUDE post-create hook used by the CPU variant).
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# Switch to root to install Python packages system-wide
USER root

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
  mv /root/.local/bin/uv /usr/local/bin/uv

# Install Python packages from requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt && \
  rm /tmp/requirements.txt

USER node

# Add PATH and PYTHONPATH to shell configs
RUN echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc && \
  echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc && \
  echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> ~/.bashrc && \
  echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> ~/.zshrc

# Configure matplotlib for VS Code plot viewer
USER root
RUN mkdir -p /home/node/.config/matplotlib
COPY .devcontainer/matplotlibrc /home/node/.config/matplotlib/matplotlibrc
RUN chown -R node:node /home/node/.config
USER node

# Copy and set up firewall script
USER root
COPY .devcontainer/init-firewall.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/init-firewall.sh && \
  echo "node ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/node-firewall && \
  chmod 0440 /etc/sudoers.d/node-firewall
USER node
