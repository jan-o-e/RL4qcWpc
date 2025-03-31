# RL4qcWpc

**Reinforcement Learning for Quantum Control with Physical Constraints**

## Installation

To set up the environment and install dependencies, follow these steps:

### Create and Activate a Virtual Environment

Using Conda:

```sh
export CONPREFIX=qiskit
conda create --prefix $CONPREFIX python=3.9 -y
conda activate $CONPREFIX
```

### Install Dependencies

Install JAX with CUDA support:

```sh
conda install -c nvidia cuda
pip install --upgrade "jax[cuda12]"
```

Install additional required packages:

```sh
pip install qiskit-dynamics gymnax evosax distrax optax flax numpy brax wandb flashbax diffrax
```

## Overview

The implementation is contained in the `rl_working` directory. Our algorithm implementation is based on the JAX-based framework [PureJAX-RL](https://github.com/luchris429/purejax-rl). We provide multiple reinforcement learning implementations:

- **Proximal Policy Optimization (PPO):**
  - `ppo_vmap_hyp.py`: PPO with hyperparameter vectorization
  - `ppo.py`: Standard PPO implementation
- **Twin Delayed Deep Deterministic Policy Gradient (TD3):** `td3.py`
- **Deep Deterministic Policy Gradient (DDPG):** `ddpg_buffer.py`

These implementations closely follow the structure of [CleanRL](https://github.com/vwxyzjn/cleanrl). There is currently a bug in td3 and ddpg_buffer which means they only run on GPUs due to dtype issues on CPUs. This will be fixed for the public release.

### Environments

Our quantum control environments are located in the `envs` directory, with support for:

- **Lambda system**
- **Rydberg atom**
- **Transmon reset**

### Reproducing Experiments & Notebooks

All experiments in our paper can be reproduced by following the structure of the example sweep provided in `rl_working/wand_sweeps`.

For quick reproducibility, we provide example Jupyter notebooks in the `notebooks` directory. These notebooks allow users to generate key results from our paper and automatically detect GPU or CPU resources for execution.

## Logging

We use **Weights & Biases (W&B)** for experiment tracking. To enable logging, configure your W&B project and entity IDs. Basic local logging is also available within the notebooks for convenience.

---