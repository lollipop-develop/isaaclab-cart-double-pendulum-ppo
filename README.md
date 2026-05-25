# Isaac Lab Cart Double Pendulum + PPO

An **underactuated cart + double pendulum** swing-up and stabilization task on **NVIDIA Isaac Lab**, solved with a small **PPO** + **GAE** implementation adapted from [nikhilbarhate99/PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch) and vectorized for Isaac Lab's parallel envs.

The system: a cart on a 1-D rail with two pole links connected in series on top. The policy commands **only** a horizontal force on the cart — the two pole joints are free. Both links start hanging straight down; the goal is to swing them both up and balance them at the upright equilibrium.

A **persistent-server** workflow lets you boot Isaac Sim once and run many `train` / `play` commands against the same loaded simulator — Ctrl+C interrupts a single command without tearing the simulator down.

This repo is the result of splitting an earlier monorepo (which also contained a single cartpole task) into two independent projects. The journey from "can't even reach upright" to "balances both links" is documented in the Notion change log; see also the sibling repo [`isaaclab-cartpole-ppo`](https://github.com/lollipop-develop/isaaclab-cartpole-ppo) for the single-pole version.

## Files

| File | Purpose |
|---|---|
| `env.py` | `DirectRLEnv` subclass with banner-marked **STATE** and **REWARD** sections |
| `ppo.py` | PPO algorithm + `HYPERPARAMS` dict (GAE, action-std schedule, etc.) |
| `server.py` | Long-running process: boots Isaac Sim once, listens on a Unix socket |
| `client.py` | Thin stdlib-only client that sends JSON commands to the server |
| `train.py` / `play.py` | Standalone runners (boot Isaac Sim per call) — kept for one-shot use |
| `Makefile` | All entry points |

## Requirements

- Linux (tested on Ubuntu 22.04)
- NVIDIA GPU + driver (validated on driver 550–580 with RTX 3090)
- Conda env with **Isaac Lab 2.3+** installed (the `isaaclab` env in this repo's setup is at `~/IsaacLab`)
- PyTorch with CUDA

This repo does *not* ship Isaac Lab itself; install it separately following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

## Quick start (recommended workflow)

### 1. Boot the persistent server (terminal A)
```bash
make env                  # GUI viewport
make env HEADLESS=1       # no GUI (faster startup, headless training)
make env NUM_ENVS=512     # more parallel envs
```

The server loads the env once for its lifetime. Wait until you see `[server] listening on .server.sock`.

### 2. Train (terminal B)
```bash
make train                                     # default: 600 iters
make train MAX_ITERS=800 RUN_NAME=swingup_v1   # named run
make train RESUME=runs/swingup_v1/policy_final.pt MAX_ITERS=400 ACTION_STD_INIT=0.15
```

Ctrl+C here kills the client; the server stays alive for the next command. Re-running with the same `RUN_NAME` auto-appends a `_HHMMSS` suffix so each run gets its own tensorboard curve and checkpoint folder.

### 3. Play a trained policy (terminal B)
```bash
make play                                                     # newest checkpoint
make play CHECKPOINT=runs/swingup_v1/policy_final.pt          # specific
make play PLAY_DET=1                                          # deterministic
```

### 4. Watch metrics
```bash
make tensorboard      # opens on http://localhost:6006
```

Five scalars are logged per training iter:
- `rollout/ep_return_mean` — main learning signal
- `rollout/ep_length_mean` — should approach 480 (full timeout) as the policy stops dying at the boundary
- `rollout/n_episodes`
- `ppo/action_std` — current Gaussian exploration noise
- `perf/steps_per_sec`

### 5. Shut the server down
- Ctrl+C in terminal A, or
- `make kill-server` from anywhere (uses SIGKILL because Isaac Sim absorbs softer signals)

## Tweaking the environment

Open `env.py` and look for the banner comments:

```python
# ============================== STATE =================================
def _get_observations(self):
    ...

# ============================== REWARD ================================
def _get_rewards(self):
    ...
```

Each section ships with the swing-up default plus a couple of commented alternatives (6-D wrap variant, sparse-only reward, etc.).

**Important:** if you change the state dim, update `observation_space` in `CartDoublePendulumEnvCfg` at the top of the same file.

After editing, restart the server (Ctrl+C → `make env`) to pick up the change. Hot reload is not supported.

## Standalone scripts (no server)

For one-shot or CI use you can bypass the server entirely:

```bash
make smoke         # 5-iter smoke test
make train-once    # equivalent to train.py
make play-once     # equivalent to play.py
```

Each call boots Isaac Sim from scratch (~30 s).

## Algorithm notes

- **PPO with GAE(λ=0.95)** — Generalized Advantage Estimation is essential on this task; vanilla MC returns produced flat curves.
- **Continuous action only** (1-D force on the cart slider). The policy outputs a `tanh`-bounded mean and adds Gaussian noise with a decaying std (`action_std_decay_*` in `ppo.HYPERPARAMS`).
- **Vectorized rollout buffer** stores tensors of shape `(T, N, …)` where T is the rollout horizon and N is `num_envs`. Discounted returns/advantages computed per-env.
- **Network 128×128** in actor and critic — bigger than the single-pole sibling because the dynamics are harder to fit.
- **No `gymnasium` install needed** — Isaac Lab's `DirectRLEnv` satisfies the interface internally; observations and rewards are torch tensors on the GPU.

## Reward function (default)

```python
r_upright              =  cos(θ₁) + cos(θ₁+θ₂)                          # in [-2, +2]
r_at_top_slow          =  both_up · (-0.1 · (θ̇₁² + θ̇₂²))                # gentle velocity damping at upright
r_stay_at_top          =  both_up · 2.0                                  # +2/step bonus at upright
r_catch                =  3.0 · exp(-dev²/0.5) · exp(-vel²/5.0)          # smooth "catch at top" reward
r_cart_center          =  -0.01 · cart_pos²                              # mild pull to center
r_cart_bound_proximity =  -0.5 · (cart_pos / max_cart_pos)⁴              # 4th-power boundary repulsion
r_terminate            =  -1.0 · terminated                              # small cart-out-of-bounds penalty
```

`both_up` triggers when both `cos(θ) > 0.95`. The `r_catch` term was the key piece for solving the stabilization sub-problem; see the Notion change log for the full reasoning.

## Layout of `runs/`

Each training run creates `runs/<run_name>/` containing:
- `policy_NNNN.pt` — periodic checkpoints (every 25 iters by default)
- `policy_final.pt` — final weights
- TensorBoard event files

Pass any of these `.pt` files to `make play CHECKPOINT=...` to roll out that snapshot.

## Acknowledgements

- PPO algorithm adapted from [nikhilbarhate99/PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch).
- Cart double pendulum asset and base `DirectRLEnv` structure from [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab).
- Sibling repo for the single cartpole version: [`isaaclab-cartpole-ppo`](https://github.com/lollipop-develop/isaaclab-cartpole-ppo).
