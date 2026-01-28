FROM ghcr.io/astral-sh/uv:latest AS uv
FROM rust:slim-bookworm

# Copy uv from official image
COPY --from=uv /uv /usr/local/bin/uv

# Install nightly toolchain, git, and build deps
RUN apt-get update && apt-get install -y \
    git build-essential && \
    rustup default nightly && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Create venv, install Python, build rmesh, install deps
RUN uv venv --python 3.12 && \
    uv pip install -e . && \
    uv pip install trimesh[all] pytest
