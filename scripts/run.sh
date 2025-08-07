#!/bin/bash
# Run script for Mini Metro RL training with comprehensive options
# Supports various training modes, logging, and monitoring

set -e

# Default values
CONFIG="dqn_config"
AGENT_TYPE=""
LOG_LEVEL="INFO"
VISUALIZE=false
HYPEROPT=false
CURRICULUM=false
TMUX_SESSION=""
TENSORBOARD=false
BACKGROUND=false
SAVE_DIR=""
EXPERIMENT_NAME=""
EPISODES=""
STEPS=""
EVAL_ONLY=false
MODEL_PATH=""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_color() {
    echo -e "${1}${2}${NC}"
}

# Print usage information
usage() {
    cat << EOF
🚇 Mini Metro RL Training Runner
================================

Usage: $0 [OPTIONS]

CONFIGURATION:
  --config CONFIG         Configuration file name (default: dqn_config)
  --agent-type AGENT      Agent type: dqn, ppo, multi
  --experiment-name NAME  Custom experiment name
  --save-dir DIR          Directory to save models

TRAINING OPTIONS:
  --episodes N            Number of training episodes
  --steps N               Number of training steps
  --hyperopt              Enable hyperparameter optimization
  --curriculum            Enable curriculum learning
  --visualize             Show real-time visualization

EVALUATION:
  --eval                  Evaluation mode only
  --model MODEL_PATH      Path to model for evaluation
  --eval-episodes N       Number of evaluation episodes (default: 100)

LOGGING & MONITORING:
  --log-level LEVEL       Logging level: DEBUG, INFO, WARNING, ERROR
  --tensorboard           Start TensorBoard server
  --tmux SESSION          Run in tmux session with given name
  --background            Run in background (daemon mode)

EXAMPLES:
  # Basic DQN training
  $0 --config dqn_config --visualize

  # PPO with hyperparameter optimization
  $0 --config ppo_config --agent-type ppo --hyperopt --steps 500000

  # Multi-agent training with curriculum learning
  $0 --config game_config --agent-type multi --curriculum --tmux multi-agent

  # Evaluation with visualization
  $0 --eval --model ./models/best_dqn.pt --agent-type dqn --visualize

  # Background training with TensorBoard
  $0 --config dqn_config --background --tensorboard --experiment-name "dqn_experiment_1"

For more information, see README.md
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --agent-type)
            AGENT_TYPE="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --eval-episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --tmux)
            TMUX_SESSION="$2"
            shift 2
            ;;
        --hyperopt)
            HYPEROPT=true
            shift
            ;;
        --curriculum)
            CURRICULUM=true
            shift
            ;;
        --visualize)
            VISUALIZE=true
            shift
            ;;
        --tensorboard)
            TENSORBOARD=true
            shift
            ;;
        --background)
            BACKGROUND=true
            shift
            ;;
        --eval)
            EVAL_ONLY=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            print_color $RED "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [ ! -f "main.py" ] || [ ! -f "train.py" ]; then
    print_color $RED "Error: Not in Mini Metro RL directory"
    print_color $YELLOW "Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_color $YELLOW "Activating Python virtual environment..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        print_color $RED "Error: Virtual environment not found"
        print_color $YELLOW "Please run setup.sh first"
        exit 1
    fi
fi

# Create directories if they don't exist
mkdir -p logs/tensorboard
mkdir -p models
mkdir -p evaluation_results

# Start TensorBoard if requested
TENSORBOARD_PID=""
if [ "$TENSORBOARD" = true ]; then
    print_color $GREEN "🔍 Starting TensorBoard server..."
    tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006 &
    TENSORBOARD_PID=$!
    print_color $BLUE "TensorBoard available at: http://localhost:6006"
    sleep 2
fi

# Function to cleanup on exit
cleanup() {
    if [ ! -z "$TENSORBOARD_PID" ]; then
        print_color $YELLOW "Stopping TensorBoard..."
        kill $TENSORBOARD_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Build command arguments
COMMON_ARGS="--config $CONFIG --log-level $LOG_LEVEL"

if [ ! -z "$AGENT_TYPE" ]; then
    COMMON_ARGS="$COMMON_ARGS --agent $AGENT_TYPE"
fi

if [ ! -z "$EXPERIMENT_NAME" ]; then
    COMMON_ARGS="$COMMON_ARGS --experiment-name $EXPERIMENT_NAME"
fi

if [ ! -z "$SAVE_DIR" ]; then
    COMMON_ARGS="$COMMON_ARGS --save-dir $SAVE_DIR"
fi

# Build command based on mode
if [ "$EVAL_ONLY" = true ]; then
    # Evaluation mode
    if [ -z "$MODEL_PATH" ]; then
        print_color $RED "Error: --model required for evaluation mode"
        exit 1
    fi
    
    EVAL_ARGS="--model $MODEL_PATH"
    if [ ! -z "$EVAL_EPISODES" ]; then
        EVAL_ARGS="$EVAL_ARGS --eval-episodes $EVAL_EPISODES"
    fi
    if [ "$VISUALIZE" = true ]; then
        EVAL_ARGS="$EVAL_ARGS --visualize"
    fi
    
    CMD="python evaluate.py $COMMON_ARGS $EVAL_ARGS --generate-report"
    print_color $GREEN "🎯 Starting evaluation..."
    
else
    # Training mode
    if [ "$HYPEROPT" = true ] && [ "$CURRICULUM" = true ]; then
        # Use train.py for advanced features
        TRAIN_ARGS=""
        if [ "$HYPEROPT" = true ]; then
            TRAIN_ARGS="$TRAIN_ARGS --hyperopt"
        fi
        if [ "$CURRICULUM" = true ]; then
            TRAIN_ARGS="$TRAIN_ARGS --curriculum"
        fi
        if [ ! -z "$STEPS" ]; then
            TRAIN_ARGS="$TRAIN_ARGS --steps $STEPS"
        fi
        
        CMD="python train.py $COMMON_ARGS $TRAIN_ARGS"
        print_color $GREEN "🧠 Starting advanced training (hyperopt/curriculum)..."
        
    else
        # Use main.py for standard training
        MAIN_ARGS="--train"
        if [ "$VISUALIZE" = true ]; then
            MAIN_ARGS="$MAIN_ARGS --visualize"
        fi
        if [ ! -z "$EPISODES" ]; then
            MAIN_ARGS="$MAIN_ARGS --episodes $EPISODES"
        fi
        if [ ! -z "$STEPS" ]; then
            MAIN_ARGS="$MAIN_ARGS --steps $STEPS"
        fi
        
        CMD="python main.py $COMMON_ARGS $MAIN_ARGS"
        print_color $GREEN "🚂 Starting training..."
    fi
fi

# Function to run command
run_command() {
    local cmd="$1"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local log_file="logs/run_${timestamp}.log"
    
    print_color $BLUE "Command: $cmd"
    print_color $BLUE "Log file: $log_file"
    
    if [ "$BACKGROUND" = true ]; then
        print_color $YELLOW "Running in background mode..."
        nohup $cmd > "$log_file" 2>&1 &
        local pid=$!
        echo $pid > "logs/training.pid"
        print_color $GREEN "Training started with PID: $pid"
        print_color $BLUE "Monitor with: tail -f $log_file"
        print_color $BLUE "Stop with: kill $pid"
    else
        # Run with tee to log and display
        $cmd 2>&1 | tee "$log_file"
    fi
}

# Function to run in tmux
run_in_tmux() {
    local session_name="$1"
    local cmd="$2"
    
    print_color $GREEN "🖥️  Starting tmux session: $session_name"
    
    # Create tmux session
    tmux new-session -d -s "$session_name"
    
    # Set up the environment in tmux
    tmux send-keys -t "$session_name" "cd $(pwd)" C-m
    tmux send-keys -t "$session_name" "source venv/bin/activate" C-m
    
    # Start TensorBoard in a separate pane if requested
    if [ "$TENSORBOARD" = true ]; then
        tmux split-window -h -t "$session_name"
        tmux send-keys -t "$session_name":0.1 "tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006" C-m
        tmux select-pane -t "$session_name":0.0
    fi
    
    # Run the main command
    tmux send-keys -t "$session_name" "$cmd" C-m
    
    # Attach to session
    print_color $BLUE "Attaching to tmux session. Use Ctrl+B then D to detach."
    tmux attach-session -t "$session_name"
}

# Display configuration
print_color $GREEN "🚇 Mini Metro RL Runner"
print_color $GREEN "======================="
echo "Configuration: $CONFIG"
echo "Agent Type: ${AGENT_TYPE:-auto-detect}"
echo "Log Level: $LOG_LEVEL"
echo "Visualize: $VISUALIZE"
echo "TensorBoard: $TENSORBOARD"
echo "Background: $BACKGROUND"
if [ ! -z "$TMUX_SESSION" ]; then
    echo "Tmux Session: $TMUX_SESSION"
fi
if [ "$EVAL_ONLY" = true ]; then
    echo "Mode: Evaluation"
    echo "Model: $MODEL_PATH"
else
    echo "Mode: Training"
    echo "Hyperopt: $HYPEROPT"
    echo "Curriculum: $CURRICULUM"
fi
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    print_color $GREEN "🔥 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
fi

# Run the command
if [ ! -z "$TMUX_SESSION" ]; then
    run_in_tmux "$TMUX_SESSION" "$CMD"
else
    run_command "$CMD"
fi

print_color $GREEN "✅ Completed successfully!"

# Show useful commands
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    print_color $BLUE "📊 Useful monitoring commands:"
    echo "  - View TensorBoard: http://localhost:6006"
    echo "  - Monitor GPU: watch -n1 nvidia-smi"
    echo "  - Check logs: tail -f logs/*.log"
    echo "  - Training service: sudo systemctl status mini-metro-rl"
fi