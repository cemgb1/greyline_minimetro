#!/bin/bash
# Monitoring script for Mini Metro RL training
# Provides comprehensive monitoring and alerting capabilities

set -e

# Configuration
REFRESH_INTERVAL=5
LOG_LINES=20
ALERT_THRESHOLD_GPU=90
ALERT_THRESHOLD_MEMORY=85
ALERT_THRESHOLD_DISK=90
EMAIL_ALERTS=false
SLACK_WEBHOOK=""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

print_color() {
    echo -e "${1}${2}${NC}"
}

print_header() {
    clear
    print_color $CYAN "🚇 Mini Metro RL Monitoring Dashboard"
    print_color $CYAN "====================================="
    echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Refresh interval: ${REFRESH_INTERVAL}s"
    echo ""
}

# Function to get system metrics
get_system_metrics() {
    # CPU usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    
    # Memory usage
    MEMORY_INFO=$(free -m | awk 'NR==2{printf "%.1f", $3*100/$2}')
    
    # Disk usage
    DISK_USAGE=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
    
    # Load average
    LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}')
    
    echo "$CPU_USAGE,$MEMORY_INFO,$DISK_USAGE,$LOAD_AVG"
}

# Function to get GPU metrics
get_gpu_metrics() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits
    else
        echo "No GPU detected"
    fi
}

# Function to get training process info
get_training_info() {
    # Check if training is running
    if [ -f "logs/training.pid" ]; then
        PID=$(cat logs/training.pid)
        if ps -p $PID > /dev/null 2>&1; then
            echo "Running (PID: $PID)"
            # Get process info
            PROC_INFO=$(ps -p $PID -o pid,ppid,%cpu,%mem,etime,cmd --no-headers)
            echo "$PROC_INFO"
        else
            echo "Not running (stale PID file)"
        fi
    else
        # Check for python processes
        PYTHON_PROCS=$(pgrep -f "python.*train.py\|python.*main.py" || echo "")
        if [ ! -z "$PYTHON_PROCS" ]; then
            echo "Running (PIDs: $PYTHON_PROCS)"
            ps -p $PYTHON_PROCS -o pid,ppid,%cpu,%mem,etime,cmd --no-headers
        else
            echo "Not running"
        fi
    fi
}

# Function to get recent logs
get_recent_logs() {
    local log_file="$1"
    local lines="$2"
    
    if [ -f "$log_file" ]; then
        tail -n "$lines" "$log_file" | while IFS= read -r line; do
            # Color code log levels
            if echo "$line" | grep -q "ERROR"; then
                print_color $RED "$line"
            elif echo "$line" | grep -q "WARNING"; then
                print_color $YELLOW "$line"
            elif echo "$line" | grep -q "INFO"; then
                print_color $GREEN "$line"
            else
                echo "$line"
            fi
        done
    else
        print_color $YELLOW "Log file not found: $log_file"
    fi
}

# Function to check TensorBoard
check_tensorboard() {
    if pgrep -f "tensorboard" > /dev/null; then
        print_color $GREEN "✓ TensorBoard running"
        TB_PID=$(pgrep -f "tensorboard")
        echo "  PID: $TB_PID"
        echo "  URL: http://localhost:6006"
    else
        print_color $YELLOW "⚠ TensorBoard not running"
    fi
}

# Function to check alerts
check_alerts() {
    local alerts=()
    
    # Parse system metrics
    IFS=',' read -r cpu_usage memory_usage disk_usage load_avg <<< "$(get_system_metrics)"
    
    # Check thresholds
    if (( $(echo "$cpu_usage > 90" | bc -l) )); then
        alerts+=("High CPU usage: ${cpu_usage}%")
    fi
    
    if (( $(echo "$memory_usage > $ALERT_THRESHOLD_MEMORY" | bc -l) )); then
        alerts+=("High memory usage: ${memory_usage}%")
    fi
    
    if (( disk_usage > ALERT_THRESHOLD_DISK )); then
        alerts+=("High disk usage: ${disk_usage}%")
    fi
    
    # Check GPU if available
    if command -v nvidia-smi &> /dev/null; then
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        if (( GPU_UTIL > ALERT_THRESHOLD_GPU )); then
            alerts+=("High GPU usage: ${GPU_UTIL}%")
        fi
    fi
    
    # Check for training errors
    if [ -f "logs/run_$(date +%Y%m%d)_*.log" ]; then
        ERROR_COUNT=$(grep -c "ERROR" logs/run_$(date +%Y%m%d)_*.log 2>/dev/null || echo "0")
        if (( ERROR_COUNT > 0 )); then
            alerts+=("Training errors detected: $ERROR_COUNT")
        fi
    fi
    
    # Display alerts
    if [ ${#alerts[@]} -gt 0 ]; then
        print_color $RED "🚨 ALERTS:"
        for alert in "${alerts[@]}"; do
            print_color $RED "  - $alert"
        done
        echo ""
    fi
}

# Function to display training metrics
show_training_metrics() {
    print_color $BLUE "📊 Training Progress:"
    
    # Look for latest log file
    LATEST_LOG=$(ls -t logs/run_*.log 2>/dev/null | head -1 || echo "")
    
    if [ ! -z "$LATEST_LOG" ]; then
        # Extract episode information
        LAST_EPISODE=$(grep "Episode" "$LATEST_LOG" | tail -1 || echo "")
        if [ ! -z "$LAST_EPISODE" ]; then
            echo "  Latest: $LAST_EPISODE"
        fi
        
        # Extract reward information
        AVG_REWARD=$(grep "avg_reward" "$LATEST_LOG" | tail -1 | grep -o "avg_reward=[0-9.-]*" | cut -d= -f2 || echo "")
        if [ ! -z "$AVG_REWARD" ]; then
            echo "  Average Reward: $AVG_REWARD"
        fi
        
        # Check for model saves
        LAST_SAVE=$(grep "Saved" "$LATEST_LOG" | tail -1 || echo "")
        if [ ! -z "$LAST_SAVE" ]; then
            echo "  $LAST_SAVE"
        fi
    else
        print_color $YELLOW "  No training logs found"
    fi
    echo ""
}

# Function to show model files
show_models() {
    print_color $PURPLE "💾 Model Files:"
    if [ -d "models" ] && [ "$(ls -A models)" ]; then
        ls -lh models/ | tail -5
    else
        print_color $YELLOW "  No models found"
    fi
    echo ""
}

# Function to show disk usage
show_disk_usage() {
    print_color $CYAN "💿 Disk Usage:"
    echo "  Project directory:"
    du -sh . 2>/dev/null || echo "  Unable to calculate"
    echo "  Models: $(du -sh models/ 2>/dev/null | cut -f1 || echo '0')"
    echo "  Logs: $(du -sh logs/ 2>/dev/null | cut -f1 || echo '0')"
    echo ""
}

# Main monitoring function
monitor() {
    while true; do
        print_header
        
        # Check alerts first
        check_alerts
        
        # System metrics
        IFS=',' read -r cpu_usage memory_usage disk_usage load_avg <<< "$(get_system_metrics)"
        print_color $GREEN "💻 System Status:"
        echo "  CPU: ${cpu_usage}%"
        echo "  Memory: ${memory_usage}%"
        echo "  Disk: ${disk_usage}%"
        echo "  Load: $load_avg"
        echo ""
        
        # GPU metrics
        if command -v nvidia-smi &> /dev/null; then
            print_color $GREEN "🔥 GPU Status:"
            GPU_METRICS=$(get_gpu_metrics)
            if [ "$GPU_METRICS" != "No GPU detected" ]; then
                echo "$GPU_METRICS" | while IFS=',' read -r util mem_used mem_total temp power_draw power_limit; do
                    echo "  GPU Utilization: ${util}%"
                    echo "  Memory: ${mem_used}MB / ${mem_total}MB ($(echo "scale=1; $mem_used*100/$mem_total" | bc)%)"
                    echo "  Temperature: ${temp}°C"
                    echo "  Power: ${power_draw}W / ${power_limit}W"
                done
            else
                print_color $YELLOW "  No GPU detected"
            fi
            echo ""
        fi
        
        # Training status
        print_color $GREEN "🚂 Training Status:"
        TRAINING_INFO=$(get_training_info)
        echo "  $TRAINING_INFO"
        echo ""
        
        # TensorBoard status
        check_tensorboard
        echo ""
        
        # Training metrics
        show_training_metrics
        
        # Model files
        show_models
        
        # Disk usage
        show_disk_usage
        
        # Recent logs
        print_color $YELLOW "📋 Recent Logs (last $LOG_LINES lines):"
        LATEST_LOG=$(ls -t logs/run_*.log 2>/dev/null | head -1 || echo "")
        if [ ! -z "$LATEST_LOG" ]; then
            get_recent_logs "$LATEST_LOG" "$LOG_LINES"
        else
            print_color $YELLOW "  No recent logs found"
        fi
        echo ""
        
        print_color $CYAN "Press Ctrl+C to exit, or wait ${REFRESH_INTERVAL}s for refresh..."
        sleep $REFRESH_INTERVAL
    done
}

# Function to show quick status
quick_status() {
    print_color $GREEN "🚇 Mini Metro RL Quick Status"
    echo "=============================="
    
    # Training status
    TRAINING_INFO=$(get_training_info)
    echo "Training: $TRAINING_INFO"
    
    # System resources
    IFS=',' read -r cpu_usage memory_usage disk_usage load_avg <<< "$(get_system_metrics)"
    echo "CPU: ${cpu_usage}% | Memory: ${memory_usage}% | Disk: ${disk_usage}%"
    
    # GPU if available
    if command -v nvidia-smi &> /dev/null; then
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        echo "GPU: ${GPU_UTIL}%"
    fi
    
    # Latest training info
    show_training_metrics
    
    # Alerts
    check_alerts
}

# Function to stop training
stop_training() {
    print_color $YELLOW "🛑 Stopping training..."
    
    if [ -f "logs/training.pid" ]; then
        PID=$(cat logs/training.pid)
        if ps -p $PID > /dev/null 2>&1; then
            kill $PID
            sleep 2
            if ps -p $PID > /dev/null 2>&1; then
                print_color $RED "Process still running, force killing..."
                kill -9 $PID
            fi
            rm -f logs/training.pid
            print_color $GREEN "✓ Training stopped"
        else
            print_color $YELLOW "Training not running"
        fi
    else
        # Try to find and kill python processes
        PYTHON_PIDS=$(pgrep -f "python.*train.py\|python.*main.py" || echo "")
        if [ ! -z "$PYTHON_PIDS" ]; then
            kill $PYTHON_PIDS
            print_color $GREEN "✓ Training processes stopped"
        else
            print_color $YELLOW "No training processes found"
        fi
    fi
}

# Parse command line arguments
case "${1:-monitor}" in
    monitor|m)
        monitor
        ;;
    status|s)
        quick_status
        ;;
    stop)
        stop_training
        ;;
    logs|l)
        LATEST_LOG=$(ls -t logs/run_*.log 2>/dev/null | head -1 || echo "")
        if [ ! -z "$LATEST_LOG" ]; then
            tail -f "$LATEST_LOG"
        else
            print_color $YELLOW "No log files found"
        fi
        ;;
    gpu)
        if command -v nvidia-smi &> /dev/null; then
            watch -n1 nvidia-smi
        else
            print_color $YELLOW "nvidia-smi not available"
        fi
        ;;
    tensorboard|tb)
        if pgrep -f "tensorboard" > /dev/null; then
            print_color $GREEN "TensorBoard already running at http://localhost:6006"
        else
            print_color $GREEN "Starting TensorBoard..."
            tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006
        fi
        ;;
    *)
        cat << EOF
🚇 Mini Metro RL Monitor

Usage: $0 [COMMAND]

COMMANDS:
    monitor, m      Full monitoring dashboard (default)
    status, s       Quick status check
    stop           Stop training processes
    logs, l        Follow latest log file
    gpu            Monitor GPU usage
    tensorboard, tb Start TensorBoard server

EXAMPLES:
    $0                # Start monitoring dashboard
    $0 status         # Quick status check
    $0 stop           # Stop training
    $0 logs           # Follow logs
EOF
        ;;
esac