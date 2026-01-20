# Dockerfile for Wan 2.1 Video Generation Homework
# Base image with CUDA support and PyTorch
# Using CUDA 12.1 base image (compatible with host CUDA 12.8 runtime)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

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
RUN pip3 install --no-cache-dir torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements first for better caching
COPY /requirements.txt /app/requirements.txt

# Create a requirements file without torch packages and flash_attn (install separately)
RUN grep -v "^torch" /app/requirements.txt | grep -v "^flash_attn" > /app/requirements_no_torch.txt || true

# Install Python dependencies
# Install huggingface_hub first, then other requirements (excluding torch and flash_attn)
RUN pip3 install --no-cache-dir huggingface_hub[cli] && \
    pip3 install --no-cache-dir -r /app/requirements_no_torch.txt

# Install flash_attn separately after torch is available
# flash_attn requires CUDA development headers - install them first
# Note: flash-attn is REQUIRED for I2V, not optional
RUN apt-get update && apt-get install -y \
    cuda-cudart-dev-12-1 \
    cuda-nvcc-12-1 \
    && rm -rf /var/lib/apt/lists/* || true

# Install flash_attn (required for I2V)
RUN pip3 install --no-cache-dir packaging ninja && \
    (pip3 install --no-cache-dir flash-attn --no-build-isolation && \
     echo "✓ flash_attn installed successfully") || \
    (echo "⚠ Warning: flash_attn installation failed." && \
     echo "  flash_attn is REQUIRED for I2V mode." && \
     echo "  You may need to install it manually inside the container." && \
     echo "  Try: pip install flash-attn --no-build-isolation" && \
     true)

# Copy the entire project
COPY . /app/



# Set environment variables
ENV PYTHONUNBUFFERED=1
# CUDA_VISIBLE_DEVICES will be set automatically by docker-entrypoint.sh
# to select GPU with highest available memory
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}

RUN pip install git+https://github.com/huggingface/diffusers
RUN pip install accelerate wand
RUN pip install -r https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/requirements.txt

RUN accelerate config default
# RUN chmod +x /app/src/run.sh
# RUN wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py -O /app/src/train_text_to_image_lora.py
# accelerate configuration saved at $HOME/.cache/huggingface/accelerate/default_config.yaml
CMD ["bash"]
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
