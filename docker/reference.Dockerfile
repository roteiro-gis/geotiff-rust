FROM rust:1.85-bookworm

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        gdal-bin \
        libtiff-tools \
        python3 \
        python3-gdal \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
