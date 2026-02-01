# CLAUDE.md - Project Context for Claude Code

## Project: Out-of-Core MOE for Multi-Task RL

### Quick Summary
Mixture of Experts architecture with tiered memory hierarchy (HBM/DRAM/NVMe) for sequential multi-task Atari learning. The hypothesis: sparse routing + memory hierarchy = implicit continual learning without catastrophic forgetting.

### Repository Structure
```
ooc_moe/
├── core/
│   ├── tiered_store.py    # TieredExpertStore - LRU caching across memory tiers
│   ├── moe_layers.py      # ExpertRouter, TieredMoELayer, MoETransformerBlock
│   └── env_detector.py    # EnvironmentDetector - predicts experts for prefetching
├── models/
│   └── moe_agent.py       # MoERLAgent, MoERLAgentConfig
├── training/
│   └── ppo_trainer.py     # PPOTrainer, SequentialGameTrainer
├── envs/
│   └── atari_wrappers.py  # DummyAtariEnv, Atari preprocessing wrappers
├── tests/
│   └── test_harness.py    # Validation tests for all components
├── run.py                 # CLI entry point
├── requirements.txt
└── README.md
```

### Current State
- ✅ Core architecture implemented and working
- ✅ Tiered storage with LRU caching and async prefetching
- ✅ MOE layers integrated with tiered storage
- ✅ Environment detector for predicting expert needs
- ✅ Full MoE RL agent (Decision Transformer style)
- ✅ PPO trainer with sequential game training
- ✅ Dummy environments for testing
- ✅ Colab notebook for cloud training
- ⏳ Real Atari training (needs `gymnasium[atari]` + `ale-py`)
- ⏳ Real NVMe integration (currently simulated)
- ⏳ Multi-GPU/multi-node support

### Key Design Decisions
1. **Emergent Specialization (Option B)**: Experts aren't pre-assigned to games. Router learns which experts work for which tasks.
2. **Prefetching**: EnvironmentDetector runs ahead of main computation to predict and prefetch experts.
3. **Simulated Latency**: For prototyping, DRAM acts as "slow storage" with artificial delays.

### How to Run

```bash
# Install dependencies
pip install torch numpy gymnasium opencv-python matplotlib

# Run tests
python run.py --mode test

# Train on dummy environments
python run.py --mode train --num_games 5 --steps_per_game 50000

# Train on Atari (requires ale-py)
pip install gymnasium[atari] ale-py
python run.py --mode train_atari --num_games 10
```

### Key Metrics to Track
- **Cache hit rate**: Target >80% during gameplay
- **Expert specialization**: Different games should use different expert cohorts
- **Retention**: Performance on old games after training new ones (forgetting test)

### Next Steps (Priority Order)

1. **Validate on Colab**: Run the notebook with more steps, verify cache hit rate improves
2. **Add proper PPO**: Current training uses simple policy gradient, need full PPO with GAE
3. **Scale experts**: Push to 256+ experts with only 32 in HBM
4. **Real Atari**: Test on actual Atari-57 games
5. **Forgetting metrics**: Implement periodic evaluation on all previous games
6. **Real NVMe**: Replace simulated storage with memory-mapped files
7. **Gradient accumulation**: Implement proper gradient handling for cold experts

### Architecture Details

**Memory Hierarchy**:
- HBM (GPU): 32 experts default, ~0ms latency
- DRAM (CPU): 128 experts default, ~1ms simulated latency  
- NVMe (Disk): All experts, ~10ms simulated latency

**Model Config (default)**:
- 256 total experts
- 512 expert dim, 2048 hidden dim
- 6 transformer layers, 8 heads
- Top-2 routing
- ~270M expert params, ~50M attention params

**Training Flow**:
1. EnvironmentDetector predicts game → prefetches expert cohort
2. Main forward pass routes tokens to top-k experts
3. TieredExpertStore fetches from fastest available tier
4. Gradients flow through, cold experts accumulate gradients
5. Periodic writeback to cold storage

### Important Files to Understand

1. `core/tiered_store.py`: The heart of the system. `TieredExpertStore` manages the memory hierarchy.
2. `core/moe_layers.py`: `TieredMoELayer.forward()` shows how routing + storage interact.
3. `core/env_detector.py`: `EnvironmentDetector.predict()` returns prefetch recommendations.
4. `models/moe_agent.py`: `MoERLAgent.forward()` orchestrates everything.

### Common Issues

- **Low cache hit rate**: Detector not learning, or HBM capacity too small
- **All experts used equally**: Router not specializing, increase aux_loss_weight
- **OOM errors**: Reduce hbm_capacity or expert_dim
- **Slow training**: Disable simulate_latency for speed tests

### Testing Changes

Always run the test harness after changes:
```bash
python run.py --mode test
```

Or run specific tests:
```python
from ooc_moe.tests.test_harness import test_tiered_store_basic, test_moe_layer
test_tiered_store_basic()
test_moe_layer()
```

### Research Questions

1. Does expert specialization emerge naturally from sequential training?
2. How does cache hit rate correlate with downstream performance?
3. Can we achieve comparable Atari scores to dense baselines with fewer active parameters?
4. Does the memory hierarchy actually reduce forgetting?

### References

- Switch Transformer (Fedus et al., 2021) - MOE architecture
- Decision Transformer (Chen et al., 2021) - RL as sequence modeling
- ZeRO-Offload (Ren et al., 2021) - Memory hierarchy for training
- Atari-57 benchmark (Bellemare et al., 2013)
