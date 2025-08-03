FROM leonrohe/cuda:10.2-devel-ubuntu18.04

# --- Environment Setup ---
ENV DEBIAN_FRONTEND=noninteractive
ENV PYOPENGL_PLATFORM=osmesa
ENV PATH=/opt/conda/envs/neucon/bin:/opt/conda/bin:$PATH
ENV CONDA_DEFAULT_ENV=neucon
ENV MAX_JOBS=2
ENV MAKEFLAGS="-j2"

WORKDIR /app

# --- System Dependencies ---
RUN apt-get update && apt-get install -y \
    wget git cmake build-essential \
    libsparsehash-dev \
    curl libgl1-mesa-glx \
    libegl1-mesa libgl1-mesa-dev \
    libosmesa6-dev libglfw3 libglfw3-dev \
    libgles2-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Miniconda Installation ---
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -ya

# --- Copy Conda Environment Files ---
COPY models/NeuralRecon/requirements.txt /app/requirements.txt
COPY models/NeuralRecon/environment.yaml /app/environment.yaml

# --- Create Conda Environment ---
RUN conda env create -f /app/environment.yaml && \
    conda clean -ya

COPY models/NeuralRecon/docker_entrypoint.sh /app/docker_entrypoint.sh
RUN chmod +x /app/docker_entrypoint.sh

# --- Final Cleanup ---
RUN conda clean -a -y && rm -rf ~/.cache/pip