FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu as build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt install -y --no-install-recommends  \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    git \
    lldb \
    procps \
    software-properties-common \
    && apt-get clean -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY install-mold.sh .

RUN ./install-mold.sh

ENV DEBIAN_FRONTEND=readline
