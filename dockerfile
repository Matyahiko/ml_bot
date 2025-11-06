FROM python:3.12-slim-bullseye AS base

# ---- 1. 時刻・プロキシまわり ------------------------------------
ARG TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG HTTP_PROXY  ARG HTTPS_PROXY
ENV http_proxy=$HTTP_PROXY  https_proxy=$HTTPS_PROXY

# ---- 2. 必要パッケージの導入 ------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential  \
    git curl wget     \
    zip unzip tmux htop valgrind \
    libssl-dev libffi-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*

# ---- 3. TA-Lib (C) のビルド --------------------------------------
ARG TALIB_VER=0.6.4
RUN wget -q https://github.com/TA-Lib/ta-lib/releases/download/v${TALIB_VER}/ta-lib-${TALIB_VER}-src.tar.gz \
    && tar -xzf ta-lib-${TALIB_VER}-src.tar.gz \
    && cd ta-lib-${TALIB_VER} \
    && ./configure --prefix=/usr \
    && make -j1 \        
    && make install \
    && cd / \
    && rm -rf ta-lib-${TALIB_VER} ta-lib-${TALIB_VER}-src.tar.gz

# ---- 4. uv と PyTorch GPU 版をインストール --------------------------
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV UV_SYSTEM=1

RUN python -m pip install --no-cache-dir --upgrade "pip>=24.0" \
    && python -m pip install --no-cache-dir uv

# PyTorch GPU 版 (CUDA 12.6) を uv 経由でインストール
RUN uv pip install --system \
        torch==2.7.1 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126

# ---- 5. アプリ配置 ----------------------------------------------
WORKDIR /app/ml_bot
COPY pyproject.toml uv.lock* ./          
RUN uv sync --system --frozen || true    
COPY . .


