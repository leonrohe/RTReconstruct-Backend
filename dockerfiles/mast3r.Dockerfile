# CUDA 12.1 base image with cuDNN and development tools
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and Python dev headers
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libusb-1.0-0 \
    libusb-1.0-0-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR \
    && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Accept Anaconda Terms of Service
RUN conda config --set always_yes true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create Conda env & install PyTorch matching CUDA version
RUN conda create -n mast3r-slam python=3.11 -y && \
    conda run -n mast3r-slam conda install -y \
        pytorch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        pytorch-cuda=12.1 -c pytorch -c nvidia

# Upgrade build tools and fix NumPy/OpenCV version mismatch
RUN conda run -n mast3r-slam pip install --upgrade pip setuptools wheel && \
    conda run -n mast3r-slam pip install "numpy>=2.0,<2.3"

# Copy the repo
COPY models/MASt3R app/models/MASt3R
COPY models/base_model.py app/models/base_model.py

# Install Python dependencies
WORKDIR /app/models/MASt3R
RUN conda run -n mast3r-slam pip install -e thirdparty/mast3r && \
    conda run -n mast3r-slam pip install -e thirdparty/in3d && \
    conda run -n mast3r-slam pip install torchcodec==0.1

# Download checkpoints
RUN mkdir -p checkpoints && \
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/ && \
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/ && \
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/

RUN chmod +x docker_entrypoint.sh
# Default command
# CMD ["conda", "run", "--no-capture-output", "-n", "mast3r-slam", "bash"]
