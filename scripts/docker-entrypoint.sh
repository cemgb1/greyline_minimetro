#!/bin/bash
# Docker entrypoint script for Mini Metro RL

set -e

# Activate virtual environment
source /app/venv/bin/activate

# Set up environment
export MINI_METRO_RL_HOME=/app
export PYTHONPATH=/app:$PYTHONPATH

# Function to wait for GPU
wait_for_gpu() {
    echo "Waiting for GPU to be available..."
    while ! nvidia-smi > /dev/null 2>&1; do
        echo "GPU not detected, waiting..."
        sleep 5
    done
    echo "GPU detected!"
    nvidia-smi
}

# Check for GPU if CUDA is requested
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "" ]] && [[ "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
    wait_for_gpu
fi

# Execute command
case "$1" in
    train)
        echo "Starting training..."
        python train.py --config "${CONFIG:-dqn_config}" "${@:2}"
        ;;
    evaluate)
        echo "Starting evaluation..."
        python evaluate.py "${@:2}"
        ;;
    main)
        echo "Starting main application..."
        python main.py "${@:2}"
        ;;
    tensorboard)
        echo "Starting TensorBoard..."
        tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006
        ;;
    jupyter)
        echo "Starting Jupyter Lab..."
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
        ;;
    bash)
        exec /bin/bash
        ;;
    *)
        echo "Running custom command: $*"
        exec "$@"
        ;;
esac