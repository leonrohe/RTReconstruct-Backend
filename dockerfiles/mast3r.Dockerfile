# Use an official NVIDIA CUDA base image (adjust CUDA version as needed)
FROM nvidia/cuda:12.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    python3.11 \
    python3.11-venv \
    python3-pip \
    python3-dev \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create and activate a Python virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools

# Clone the repository (or assume source is mounted under /workspace)
WORKDIR /workspace
# Uncomment below to clone directly:
# RUN git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git .
# If building with local files: COPY . /workspace

# Install MASt3R-SLAM and submodules
RUN pip install --no-cache-dir -e thirdparty/mast3r \
 && pip install --no-cache-dir -e thirdparty/in3d \
 && pip install --no-cache-dir --no-build-isolation -e .

# Optional: Install torchcodec for faster mp4 loading
RUN pip install --no-cache-dir torchcodec==0.1 || echo "torchcodec install failed, proceeding without it"

# Prepare checkpoints
RUN mkdir -p checkpoints \
 && wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/ \
 && wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/ \
 && wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/

# Expose necessary ports if exposing services later
# EXPOSE 8888

# Default working directory
WORKDIR /workspace

# Default command (can be overridden)
CMD ["python", "main.py", "--help"]
