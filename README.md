# Outdated and probably wrong readme

## Current setup and training (up to date)

```bash
# Installs the right PyTorch for this machine (NVIDIA/CUDA, AMD/ROCm or CPU)
# plus the project dependencies. Keeps an already-working GPU torch.
scripts/setup_gpu.sh            # or: scripts/setup_gpu.sh cuda|rocm|cpu
# On vast.ai images that already ship torch: USE_SYSTEM_PYTHON=1 scripts/setup_gpu.sh

# Train: rollouts run on CPU worker processes (all cores but one by default),
# the PPO update runs on the GPU with bf16/fp16 mixed precision.
.venv/bin/python -m clasher.rl.train_selfplay --rollout-steps 4096 --quiet-engine

# Watch a checkpoint (policy = blue, bottom) / hand-train it
.venv/bin/python -m clasher.rl.watch_policy_battle --checkpoint <ckpt.pt> --opponent-random
.venv/bin/python -m training.imitation.handTrain --checkpoint <ckpt.pt>
```

Do not run `uv sync`: torch is not in `uv.lock` (the build depends on the
GPU vendor), so sync would uninstall it. Use `uv pip install ...` instead.


# 🏆 Clash Royale Battle Engine

A blazing-fast, feature-complete Clash Royale battle simulation engine built in Python. Designed for reinforcement learning research and game AI development.

## ✨ Features

### 🚀 Core Engine
- **Authentic mechanics** from official gamedata.json
- **33ms fixed timestep** (30 FPS) for deterministic simulation
- **732k+ ticks/second** performance in turbo mode
- **Bridge pathfinding** and proper troop movement
- **Complete spell system** (Arrows, Fireball, Zap, Lightning)
- **Win conditions** with crown counting and overtime

### 🤖 Machine Learning Ready
- **Gymnasium environment** compatible with Stable-Baselines3, RLlib
- **128×128×3 observation tensor** (owner mask, troop type, HP)
- **2304 discrete actions** encoding (card_idx, x_tile, y_tile)  
- **Reward system** based on crowns, tower damage, victories
- **Batch simulation** for distributed training

### 🎮 Visualization & Analysis
- **Real-time pygame visualizer** with health bars and UI
- **Replay recording** with msgspec for fast serialization
- **Battle statistics** and performance benchmarking
- **Advanced mechanics** (knockback, stun, death spawns)

## 🏃 Quick Start

### Basic Battle
```python
from src.clasher.engine import BattleEngine
from src.clasher.arena import Position

# Create and run a battle
engine = BattleEngine("gamedata.json")
battle = engine.create_battle()

# Deploy some cards
battle.deploy_card(0, "Knight", Position(16.0, 8.0))
battle.deploy_card(1, "Archers", Position(16.0, 10.0))

# Run simulation
for _ in range(1000):
    battle.step()
    if battle.game_over:
        break

print(f"Winner: Player {battle.winner}")
```

### Visualized Battle
```python
python3 visualized_battle.py
```
Watch battles unfold in real-time with the pygame visualizer!

### RL Training
```python
from src.clasher.gym_env import ClashRoyaleGymEnv

env = ClashRoyaleGymEnv()
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Your RL agent here
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### PPO Self-Play (Fast Path)
```bash
# uv-managed env
uv sync

# CPU rollout workers + MPS learner (stable default)
PYTHONPATH=src uv run python -m clasher.rl.train_selfplay \
  --num-workers 6 \
  --device mps \
  --quiet-engine \
  --rollout-steps 768 \
  --save-every 5 \
  --checkpoint-dir checkpoints/selfplay_run \
  --resume-latest

# Async actors + learner overlap (higher throughput)
PYTHONPATH=src uv run python -m clasher.rl.train_selfplay_async \
  --num-actors 6 \
  --device mps \
  --quiet-engine \
  --actor-rollout-steps 128 \
  --transitions-per-update 3072 \
  --batch-size 1024 \
  --save-every 5 \
  --checkpoint-dir checkpoints/selfplay_async \
  --resume-latest
```

### Watch Latest Policy
```bash
CKPT=$(ls -1t checkpoints/selfplay_run/policy_update_*.pt | head -n 1)
PYTHONPATH=src uv run python -m clasher.rl.watch_policy_battle \
  --checkpoint "$CKPT" \
  --device mps
```

### Turbo Benchmarking  
```python
from src.clasher.replay import TurboEngine

turbo = TurboEngine("gamedata.json")
results = turbo.benchmark(duration_seconds=10)
print(f"Speed: {results['ticks_per_second']:,} ticks/sec")
```

## 📋 Installation

```bash
# Clone and install dependencies
git clone <your-repo>
cd clasher
pip3 install -r requirements.txt

# Run tests
python3 -m pytest tests/ -v

# Try the demos
python3 example.py                 # Basic engine demo
python3 advanced_demo.py          # Advanced features
python3 test_gym_env.py           # RL environment test
python3 visualized_battle.py      # Visual battle
python3 final_showcase.py         # Complete showcase
```

## 🏗️ Architecture

```
src/clasher/
├── data.py              # Card data loading from gamedata.json
├── arena.py            # Arena geometry and positioning  
├── entities.py         # Troops, Buildings, Projectiles
├── player.py           # Player state and hand management
├── battle.py           # Core battle simulation
├── engine.py           # High-level battle engine
├── spells.py           # Spell system and effects
├── gym_env.py          # Gymnasium RL environment
├── replay.py           # Replay recording and turbo mode
├── visualizer.py       # Pygame real-time visualizer
└── advanced_mechanics.py # Knockback, stun, death spawns
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Simulation speed** | 732k+ ticks/sec |
| **Real-time speedup** | 24,000x faster |
| **Battles/second** | 67+ sustained |
| **Cards supported** | 146 from gamedata.json |
| **Memory usage** | ~50MB per battle |

## 🧪 Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Specific test suites
python3 -m pytest tests/test_basic.py      # Core functionality
python3 -m pytest tests/test_advanced.py  # Advanced features
```

**Test Coverage:**
- ✅ Card deployment and elixir system
- ✅ Knight vs Knight mid-bridge battles
- ✅ Win condition detection  
- ✅ Spell casting and effects
- ✅ Turbo mode performance
- ✅ RL environment integration

## 🤖 ML Integration Examples

### Stable-Baselines3
```python
from stable_baselines3 import PPO
from src.clasher.gym_env import ClashRoyaleGymEnv

env = ClashRoyaleGymEnv(speed_factor=10.0)
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### Custom Training Loop
```python
from src.clasher.replay import TurboEngine

turbo = TurboEngine("gamedata.json")
results = turbo.run_batch(num_battles=1000, record_replays=True)

# Analyze results for training data
for result in results:
    battle_data = result["result"]
    # Extract features, rewards, etc.
```

## 🎯 Roadmap

### Implemented ✅
- All 9 phases from plan.md complete
- Core battle mechanics and physics
- Gymnasium RL environment
- Real-time visualization
- Advanced mechanics system

### Future Enhancements 🚧
- More cards and mechanics from gamedata.json  
- Tournament and ladder systems
- Distributed training support
- WebGL/browser visualization
- Multi-agent training scenarios

## 📚 References

Built following the comprehensive plan.md roadmap, incorporating insights from:
- [RetroRoyale](https://github.com/retroroyale/ClashRoyale) - .NET server implementation
- [Build-A-Bot](https://github.com/Pbatch/ClashRoyaleBuildABot) - Computer vision bot
- [clash-royale-gym](https://github.com/MSU-AI/clash-royale-gym) - RL environment
- [RoyaleAPI](https://github.com/RoyaleAPI/cr-api-data) - Official game data

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features  
4. Ensure all tests pass
5. Submit a pull request

---

**🏆 Ready for production ML training and game AI research!** 🤖
