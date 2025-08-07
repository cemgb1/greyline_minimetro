# 🚇 Mini Metro Reinforcement Learning

A comprehensive, production-ready reinforcement learning implementation for the Mini Metro game with maximum accuracy and extensive logging capabilities.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

This project implements a high-fidelity Mini Metro game simulation with advanced reinforcement learning agents. Features include accurate game mechanics, multiple RL algorithms (DQN, PPO, Multi-agent), comprehensive visualization, and production-ready deployment capabilities.

### Key Features

- **🎮 High-Fidelity Game Simulation**: Accurate Mini Metro mechanics with real-time train movement, passenger spawning, and station management
- **🧠 Advanced RL Agents**: DQN, PPO, and Multi-agent implementations with modern improvements
- **📊 Comprehensive Logging**: TensorBoard integration with custom metrics, network visualization, and performance analysis
- **🎨 Real-time Visualization**: Pygame-based renderer with smooth animations and interactive controls
- **☁️ GCP Ready**: Complete deployment scripts and Docker containers for cloud training
- **🔧 Production Features**: Hyperparameter optimization, curriculum learning, and comprehensive monitoring

## 🏗️ Architecture

```
mini_metro_rl/
├── src/
│   ├── game/              # Core game mechanics
│   │   ├── mini_metro_game.py    # Main game engine
│   │   ├── station.py            # Station entities with types and queues
│   │   ├── train.py              # Train movement and capacity
│   │   ├── line.py               # Metro line management
│   │   └── passenger.py          # Passenger behavior and satisfaction
│   ├── environment/       # RL environment
│   │   ├── metro_env.py          # Gymnasium environment
│   │   └── rewards.py            # Multi-objective reward system
│   ├── agents/           # RL agents
│   │   ├── dqn_agent.py         # Deep Q-Network with improvements
│   │   ├── ppo_agent.py         # Proximal Policy Optimization
│   │   └── multi_agent.py       # Multi-agent coordination
│   ├── visualization/    # Rendering and logging
│   │   ├── pygame_renderer.py   # Real-time game visualization
│   │   └── tensorboard_logger.py # Comprehensive logging
│   └── utils/           # Utilities and configuration
│       ├── config.py            # YAML configuration management
│       └── helpers.py           # Common utilities
├── configs/             # Configuration files
├── scripts/            # Deployment and monitoring scripts
├── main.py            # Main entry point
├── train.py          # Advanced training script
└── evaluate.py       # Comprehensive evaluation
```

## 🚀 Quick Start

### Local Installation

1. **Clone the repository:**
```bash
git clone https://github.com/cemgb1/greyline_minimetro.git
cd greyline_minimetro
```

2. **Run the setup script:**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

3. **Activate the environment:**
```bash
source ~/.bashrc
metro-rl  # Activates virtual environment and navigates to project
```

4. **Start training:**
```bash
# Basic DQN training with visualization
python main.py --train --visualize --config dqn_config

# Advanced training with hyperparameter optimization
python train.py --config dqn_config --hyperopt --tensorboard

# Multi-agent training
python main.py --train --agent multi --config game_config
```

### Docker Deployment

```bash
# Build the image
docker build -t mini-metro-rl .

# Run training
docker run --gpus all -v $(pwd)/models:/app/models mini-metro-rl train

# Run with TensorBoard
docker run --gpus all -p 6006:6006 mini-metro-rl tensorboard

# Interactive development
docker run --gpus all -it -p 8888:8888 mini-metro-rl jupyter
```

### GCP Deployment

1. **Create GCP instance:**
```bash
gcloud compute instances create mini-metro-rl \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE \
    --metadata-from-file startup-script=scripts/setup.sh
```

2. **SSH and start training:**
```bash
gcloud compute ssh mini-metro-rl --zone=us-central1-a
cd mini_metro_rl
./scripts/run.sh --config dqn_config --tensorboard --background
```

3. **Monitor training:**
```bash
./scripts/monitor.sh
```

## 🎮 Game Features

### Accurate Mini Metro Mechanics

- **Station Types**: Circle, Triangle, Square, Pentagon, Hexagon, Diamond, Star, Cross
- **Train Systems**: Real-time movement, capacity management, carriage upgrades
- **Passenger Behavior**: Realistic spawning patterns, satisfaction tracking, transfer management
- **Line Management**: Bidirectional/loop configurations, bridge/tunnel mechanics
- **Weekly Progression**: Escalating difficulty with passenger spawn rate increases

### Advanced State Representation

The environment provides comprehensive state information:
- Station positions, types, queue lengths, and capacity utilization
- Real-time train positions, directions, passenger loads, and destinations
- Line configurations, connectivity, and performance metrics
- Available resources, passenger flow patterns, and topology metrics
- Time-based progression and performance indicators

### Complete Action Space

Agents can perform all major Mini Metro actions:
- Create and extend metro lines
- Add trains and carriages to lines
- Convert between linear and loop configurations
- Add bridges and tunnels for water crossings
- Manage station connections and line priorities

## 🧠 RL Agents

### DQN Agent
Advanced Deep Q-Network implementation with:
- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Experience Replay**: Focuses on important transitions
- **Noisy Networks**: Parameter space exploration
- **Multi-step Learning**: Improved sample efficiency

### PPO Agent
Proximal Policy Optimization with:
- **Actor-Critic Architecture**: Shared feature extraction
- **Generalized Advantage Estimation (GAE)**: Reduced variance
- **Clipped Objective**: Stable policy updates
- **Value Function Clipping**: Additional stability
- **Entropy Regularization**: Exploration encouragement

### Multi-Agent System
Sophisticated multi-agent framework featuring:
- **Line-based Agents**: Each metro line controlled by separate agent
- **Communication Protocols**: Agent coordination mechanisms
- **Shared Experience**: Optional experience sharing between agents
- **Centralized Critic**: Centralized training with decentralized execution
- **Dynamic Agent Management**: Create/remove agents as lines are built

## 📊 Visualization & Logging

### Real-time Pygame Visualization
- **Smooth Animations**: Interpolated train movement and state transitions
- **Interactive Controls**: Pause, step, speed control, and camera movement
- **Station Visualization**: Queue indicators, types, and performance metrics
- **Line Rendering**: Proper colors, topology, and bridge/tunnel indicators
- **Performance Overlay**: Real-time metrics and game state information

### Comprehensive TensorBoard Logging
- **Training Metrics**: Loss curves, reward progression, and convergence analysis
- **Game-specific Metrics**: Passenger delivery rates, satisfaction scores, network efficiency
- **Network Visualization**: Topology graphs, connectivity analysis, and bottleneck identification
- **Q-value Heatmaps**: Action preference visualization and policy interpretation
- **Custom Dashboards**: Real-time monitoring and experiment comparison

## ⚙️ Configuration

The project uses YAML-based configuration with modular components:

```yaml
# Example DQN configuration
experiment_name: "dqn_baseline"
game:
  map_name: "london"
  difficulty: "normal"
  max_episode_steps: 3000

dqn:
  learning_rate: 0.0001
  gamma: 0.99
  buffer_size: 1000000
  network:
    hidden_sizes: [512, 512, 256]
    activation: "relu"
  double_dqn: true
  dueling_dqn: true

training:
  total_timesteps: 1000000
  tensorboard_log: true
  save_frequency: 50000
```

## 🔧 Advanced Features

### Hyperparameter Optimization
Integrated Optuna-based hyperparameter optimization:
```bash
python train.py --config dqn_config --hyperopt --trials 100
```

### Curriculum Learning
Progressive difficulty training:
```bash
python train.py --config game_config --curriculum
```

### Performance Monitoring
Comprehensive monitoring with alerts:
```bash
./scripts/monitor.sh  # Full dashboard
./scripts/monitor.sh status  # Quick status
./scripts/monitor.sh gpu  # GPU monitoring
```

## 📈 Evaluation & Analysis

### Comprehensive Evaluation
```bash
# Evaluate single model
python evaluate.py --model models/best_dqn.pt --agent-type dqn --episodes 100 --visualize

# Compare multiple models
python evaluate.py --model models/ --generate-report

# Statistical analysis with visualizations
python evaluate.py --model models/best_dqn.pt --save-episodes --generate-report
```

### Performance Metrics
- **Reward Analysis**: Distribution, trends, and statistical significance
- **Game Performance**: Passenger delivery rates, satisfaction scores, survival time
- **Network Efficiency**: Connectivity metrics, resource utilization, bottleneck analysis
- **Agent Behavior**: Action distributions, Q-value analysis, policy interpretation

## 🐳 Docker & Deployment

### Multi-stage Docker Build
```dockerfile
# Development with Jupyter
docker build --target jupyter -t mini-metro-rl:dev .

# Production training
docker build --target training -t mini-metro-rl:train .

# GPU-optimized
docker build --target gpu-production -t mini-metro-rl:gpu .
```

### GCP Best Practices
- **Preemptible Instances**: Cost-effective training with checkpointing
- **Cloud Logging**: Centralized log management and monitoring
- **Cloud Storage**: Model artifacts and experiment data
- **Monitoring**: Custom dashboards and alerting

## 🧪 Testing

Run the test suite:
```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/test_integration.py -v

# Performance tests
pytest tests/test_performance.py -v --benchmark-only
```

## 📚 Documentation

Detailed documentation is available:
- [API Reference](docs/api.md)
- [Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original Mini Metro game by Dinosaur Polo Club
- OpenAI Gymnasium framework
- PyTorch deep learning library
- Stable Baselines3 for RL implementations
- The reinforcement learning community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/cemgb1/greyline_minimetro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cemgb1/greyline_minimetro/discussions)
- **Documentation**: [Project Wiki](https://github.com/cemgb1/greyline_minimetro/wiki)

## 🔮 Roadmap

- [ ] Additional RL algorithms (A3C, SAC, TD3)
- [ ] Hierarchical reinforcement learning
- [ ] Real-world transit data integration
- [ ] Multi-objective optimization
- [ ] Web-based visualization dashboard
- [ ] Mobile deployment capabilities

---

**Built with ❤️ for the reinforcement learning and transit optimization community**