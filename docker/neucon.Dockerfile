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

# --- Manually Reinstall TorchSparse to Ensure CUDA Extensions Are Built ---
RUN pip install --force-reinstall \
    git+https://github.com/mit-han-lab/torchsparse.git@48cc7e23784d35e67163f61b9312df138853025e && \
    python -c "import torchsparse.backend as backend; assert hasattr(backend, 'hash_cuda'), 'hash_cuda not found!'"

# --- Download Model Checkpoint ---
RUN mkdir -p /app/models/NeuralRecon/checkpoints && \
    conda run -n neucon gdown --id 1zKuWqm9weHSm98SZKld1PbEddgLOQkQV -O /app/models/NeuralRecon/checkpoints/model.ckpt

# --- Final Cleanup ---
RUN conda clean -a -y && rm -rf ~/.cache/pip