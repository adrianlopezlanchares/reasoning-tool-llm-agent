# Dockerfile for Wan 2.1 Video Generation Homework
# Base image with CUDA support and PyTorch
# Using CUDA 12.1 base image (compatible with host CUDA 12.8 runtime)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libx264-dev \
    libxvidcore-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip, setuptools, and wheel to latest versions
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch 2.4.0+ with CUDA 12.1 support (compatible with CUDA 12.8 runtime)
# CUDA 12.1 wheels work with CUDA 12.8 runtime due to backward compatibility
COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["bash"]