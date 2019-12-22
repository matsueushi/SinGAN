# Dockerfile Julia 1.3.0

ARG CUDA=10.0
ARG UBUNTU_VERSION=18.04

FROM nvidia/cuda:${CUDA}-cudnn7-devel-ubuntu${UBUNTU_VERSION}

ENV JULIA_PATH=/usr/local/julia
ENV PATH=$JULIA_PATH/bin:$PATH

ENV JULIA_TAR_ARCH=x86_64
ENV JULIA_DIR_ARCH=x64

ENV JULIA_GPG=3673DF529D9049477F76B37566E3C7DC03D6E495
ENV JULIA_VERSION=1.3.0
ENV JULIA_SHA256=9ec9e8076f65bef9ba1fb3c58037743c5abb3b53d845b827e44a37e7bcacffe8

# Based on https://github.com/docker-library/julia
# Copyright (c) 2014 Docker, Inc.
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl gnupg dirmngr; \
    rm -rf /var/lib/apt/lists/*; \
    \
    folder="$(echo "$JULIA_VERSION" | cut -d. -f1-2)"; \
    julia_tar_url="https://julialang-s3.julialang.org/bin/linux/${JULIA_DIR_ARCH}/${folder}/julia-${JULIA_VERSION}-linux-${JULIA_TAR_ARCH}.tar.gz"; \
    curl -fL -o julia.tar.gz.asc "${julia_tar_url}.asc"; \
    curl -fL -o julia.tar.gz     "${julia_tar_url}"; \
    \
    echo "${JULIA_SHA256} *julia.tar.gz" | sha256sum -c -; \
    \
    export GNUPGHOME="$(mktemp -d)"; \
    gpg --batch --keyserver ha.pool.sks-keyservers.net --recv-keys "$JULIA_GPG"; \
    gpg --batch --verify julia.tar.gz.asc julia.tar.gz; \
    command -v gpgconf > /dev/null && gpgconf --kill all; \
    rm -rf "$GNUPGHOME" julia.tar.gz.asc; \
    \
    mkdir "$JULIA_PATH"; \
    tar -xzf julia.tar.gz -C "$JULIA_PATH" --strip-components 1; \
    rm julia.tar.gz; \
    \
    # smoke test
    julia --version