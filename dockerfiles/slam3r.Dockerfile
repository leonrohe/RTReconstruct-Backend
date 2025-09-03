# Base image with CUDA 11.8 and Python 3.11
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget unzip ffmpeg libgl1-mesa-glx \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set up virtual environment
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

COPY models/SLAM3R/requirements.txt /app/requirements.txt
COPY models/SLAM3R/requirements_optional.txt /app/requirements_optional.txt

# Upgrade pip and install core dependencies
RUN pip install --upgrade pip && \
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt && \
    pip install -r requirements_optional.txt && \
    pip install xformers==0.0.28.post2

# COPY models/SLAM3R/docker_entrypoint.sh /app/docker_entrypoint.sh
# RUN chmod +x /app/docker_entrypoint.sh

# # Compile RoPE CUDA kernels
# RUN cd models/SLAM3R/slam3r/pos_embed/curope && \
#     python setup.py build_ext --inplace && \
#     cd ../../../../

# Expose a port for the Gradio interface
# EXPOSE 7860

# Default command
# CMD ["/workspace/SLAM3R/venv/bin/python", "app.py", "--device", "cuda", "--local_network", "--server_port", "7860"]
