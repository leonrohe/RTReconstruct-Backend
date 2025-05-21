FROM leonrohe/cuda:10.2-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYOPENGL_PLATFORM=osmesa
ENV PATH=/opt/conda/bin:$PATH
ENV MAX_JOBS=2
ENV MAKEFLAGS="-j2"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget git cmake build-essential \
    libsparsehash-dev \
    curl libgl1-mesa-glx \
    libegl1-mesa libgl1-mesa-dev \
    libosmesa6-dev libglfw3 libglfw3-dev \
    libgles2-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -ya

# Copy only environment.yaml for env creation (assumes it's in models/neucon/NeuralRecon)
COPY models/NeuralRecon/requirements.txt /app/requirements.txt
COPY models/NeuralRecon/environment.yaml /app/environment.yaml

# Create conda environment
RUN conda env create -f /app/environment.yaml && \
    conda clean -ya

# Download model checkpoint using conda environment
RUN mkdir -p /app/models/NeuralRecon/checkpoints && \
    conda run -n neucon gdown --id 1zKuWqm9weHSm98SZKld1PbEddgLOQkQV -O /app/models/NeuralRecon/checkpoints/model.ckpt
