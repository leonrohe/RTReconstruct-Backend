# Stage 1: Base CUDA development image
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# Install dependencies including nvidia-utils
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
RUN python3 -m pip install --break-system-packages --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000