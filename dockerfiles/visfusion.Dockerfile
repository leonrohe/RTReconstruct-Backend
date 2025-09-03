FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

WORKDIR /app

# --- System Dependencies ---
ENV DEBIAN_FRONTEND=noninteractive
ENV PYOPENGL_PLATFORM=osmesa
ENV MAX_JOBS=2
ENV MAKEFLAGS="-j2"

RUN apt-get update && apt-get install -y \
    wget git cmake build-essential \
    libsparsehash-dev \
    curl libgl1-mesa-glx \
    libegl1-mesa libgl1-mesa-dev \
    libosmesa6-dev libglfw3 libglfw3-dev \
    libgles2-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev \
    python3-distutils && \
    rm -rf /var/lib/apt/lists/*

# --- Miniconda Installation ---
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

# Make conda available in PATH
ENV PATH=/opt/conda/bin:$PATH

# Accept Anaconda Terms of Service
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# --- Copy Conda Environment Files ---
COPY models/VisFusion/environment.yml /app/environment.yml
COPY models/VisFusion/docker_entrypoint.sh /app/docker_entrypoint.sh
RUN chmod +x /app/docker_entrypoint.sh

# --- Create Conda Environment ---
RUN conda env create -f /app/environment.yml

# --- Environment Setup (AFTER env is created) ---
ENV PATH=/opt/conda/envs/visfusion/bin:/opt/conda/bin:$PATH
ENV CONDA_DEFAULT_ENV=visfusion

# --- Final Cleanup ---
RUN conda clean -a -y && rm -rf ~/.cache/pip
