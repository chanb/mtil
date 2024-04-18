# Multitask Imitation Learning Experiments

## Preliminary Experiments
- PPO Objective Comparison: We compare PPO objectives in discrete action space: (1) Clipping objective and (2) Reverse-KL objective
- Necessity of Finetuning in Pendulum: We compare the performance change in transferring an expert policy from a source environment to target environments
- Behavioural Cloning (BC) Ablations
    - The amount of demonstration data required
    - The impact of subsampling scheme

## Main Experiments
- Amount of source tasks and source data
- Amount of target data

# Directory Structure
We assume the working directory is the directory containing this README:
```
.
├── ablations/
│   ├── bc_amount_data/
│   ├── bc_subsampling/
│   ├── expert_policies/
│   ├── main/
│   ├── metrics/
│   ├── objective_comparison/
│   ├── policy_robustness_pendulum/
│   └── utils/
├── experiments/
├── main/
│   ├── main/
│   └── metrics/
└── README.md
```

- The `experiments` directory contains the configuration files for running the experiments.
In other words, we use the containing `json` files when calling `python main.py`.
- The `ablations` directory contains (1) expert policies, (2) various ablation experiment scripts, and (3) plotting scripts.
- The `main` directory contains the plotting script for the main experiment and the task diversity analysis.
