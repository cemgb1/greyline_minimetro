# Multi-stage Docker build for Mini Metro RL
# Optimized for both development and production deployment

FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev \
    ffmpeg \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash metro && \
    chown -R metro:metro /app

# Switch to non-root user
USER metro

# Create Python virtual environment
RUN python3.9 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Development stage
FROM base as development

# Copy requirements first for better caching
COPY --chown=metro:metro requirements.txt .
RUN pip install -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional development tools
RUN pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    black \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pre-commit

# Copy project files
COPY --chown=metro:metro . .

# Create necessary directories
RUN mkdir -p logs/tensorboard models data evaluation_results checkpoints

# Expose ports
EXPOSE 8888 6006 8000

# Default command for development
CMD ["bash"]

# Production stage
FROM base as production

# Copy requirements and install
COPY --chown=metro:metro requirements.txt .
RUN pip install -r requirements.txt && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy only necessary files for production
COPY --chown=metro:metro mini_metro_rl/ ./mini_metro_rl/
COPY --chown=metro:metro configs/ ./configs/
COPY --chown=metro:metro scripts/ ./scripts/
COPY --chown=metro:metro main.py train.py evaluate.py ./

# Create directories
RUN mkdir -p logs/tensorboard models data evaluation_results checkpoints

# Make scripts executable
RUN chmod +x scripts/*.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('GPU available:', torch.cuda.is_available())" || exit 1

# Expose TensorBoard port
EXPOSE 6006

# Default command for production
CMD ["python", "train.py", "--config", "dqn_config"]

# GPU-optimized production stage
FROM production as gpu-production

# Install additional GPU monitoring tools
USER root
RUN apt-get update && apt-get install -y \
    nvidia-utils-470 \
    && rm -rf /var/lib/apt/lists/*

USER metro

# Set GPU-specific environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Training stage - optimized for long-running training jobs
FROM production as training

# Install additional monitoring and debugging tools
RUN pip install \
    wandb \
    optuna \
    ray[tune] \
    psutil \
    gpustat

# Create training-specific directories
RUN mkdir -p \
    experiments \
    hyperopt_results \
    curriculum_stages \
    model_checkpoints

# Copy training scripts
COPY --chown=metro:metro scripts/run.sh scripts/monitor.sh ./scripts/

# Set environment variables for training
ENV MINI_METRO_RL_HOME=/app
ENV PYTHONPATH=/app:$PYTHONPATH
ENV CUDA_LAUNCH_BLOCKING=1

# Default training command
CMD ["bash", "scripts/run.sh", "--config", "dqn_config", "--tensorboard"]

# Evaluation stage - optimized for model evaluation and testing
FROM production as evaluation

# Install additional visualization tools
RUN pip install \
    plotly \
    seaborn \
    dash \
    streamlit

# Create evaluation-specific directories
RUN mkdir -p \
    evaluation_reports \
    comparison_results \
    benchmark_data

# Default evaluation command
CMD ["python", "evaluate.py", "--help"]

# Multi-agent stage - optimized for multi-agent experiments
FROM training as multi-agent

# Install additional multi-agent libraries
RUN pip install \
    ray[rllib] \
    stable-baselines3[extra] \
    pettingzoo \
    supersuit

# Set multi-agent specific environment
ENV RAY_DISABLE_IMPORT_WARNING=1

# Default multi-agent command
CMD ["python", "train.py", "--config", "game_config", "--agent-type", "multi"]

# Jupyter stage - for interactive development and analysis
FROM development as jupyter

# Install additional Jupyter extensions
RUN pip install \
    jupyter-dash \
    plotly \
    bokeh \
    altair \
    nbconvert \
    papermill

# Configure Jupyter
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> ~/.jupyter/jupyter_lab_config.py

# Copy example notebooks
COPY --chown=metro:metro notebooks/ ./notebooks/

# Expose Jupyter port
EXPOSE 8888

# Default Jupyter command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Final production image
FROM production as final

# Add labels for image metadata
LABEL maintainer="Mini Metro RL Team"
LABEL version="1.0.0"
LABEL description="Mini Metro Reinforcement Learning Environment"
LABEL gpu.required="true"

# Add startup script
COPY --chown=metro:metro scripts/docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Use entrypoint script
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["train"]