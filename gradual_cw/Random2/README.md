## Brain-Inspired Warm-Up Training with Random Noise for Uncertainty Calibration

Jeonghwan Cheon*, Se-Bum Paik† 

\* First author: jeonghwan518@kaist.ac.kr  
† Corresponding author: sbpaik@kaist.ac.kr  


### Associated article

This repository contains the implementation and demo codes for the paper "**Brain-Inspired Warm-Up Training with Random Noise for Uncertainty Calibration**" (Cheon and Paik, Nature Machine Intelligence, 2026). The preprint is available on arXiv.

[![arXiv](https://img.shields.io/badge/arXiv-2412.17411-b31b1b.svg)](https://arxiv.org/abs/2412.17411)

#### Abstract

Uncertainty calibration, the alignment of predictive confidence with accuracy, is essential for the reliable deployment of machine learning systems in real-world applications. However, current models often fail to achieve this goal, generating responses that are overconfident, inaccurate or even fabricated. Here, we show that the widely adopted initialization method in deep learning, long regarded as standard practice, is in fact a primary source of overconfidence. To address this problem, we introduce a neurodevelopment-inspired warm-up strategy that inherently resolves uncertainty-related issues without requiring pre- or post-processing. In our approach, networks are first briefly trained on random noise and random labels before being exposed to real data. This warm-up phase yields optimal calibration, ensuring that confidence remains well aligned with accuracy throughout subsequent training. Moreover, the resulting networks demonstrate high proficiency in the identification of “unknown” inputs, providing a robust solution for uncertainty calibration in both in-distribution and out-of-distribution contexts.

## Repository Layout

```text
.
├── cli/      # Shell wrappers for end-to-end experiment execution
├── scripts/  # Python entry points for training/evaluation experiments
├── src/      # Core library code (models, training, data, evaluation, logging)
├── custom/   # Custom plotting/font utilities
├── data/     # Dataset download/cache location
├── source_data/ # Source JSON/stats and regeneration scripts for paper figures/tables
└── exp/      # Experiment outputs (JSON, checkpoints, figures)
```

## Environment Setup

Recommended Python version: `3.12.x`  
Supported range (from `pyproject.toml`): `>=3.12,<3.13`.

Option A (recommended): `uv`

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

Option B: `pip` + `requirements.txt` (pinned, reproducible)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Option C: `pip` + `requirements.in` (top-level dependencies)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.in
```

The CLI wrappers and examples assume `python` is available in `PATH`; activating `.venv` provides this.

## Main Experiments

Run training + calibration:

```bash
./cli/run_experiment_calibration.sh
```

Run training + OOD detection:

```bash
./cli/run_experiment_ood_detection.sh
```

Argument split markers:
- `--calib-args`: arguments before this go to `experiment_training.py`; arguments after this go to `experiment_calibration.py`.
- `--ood-args`: arguments before this go to `experiment_training.py`; arguments after this go to `experiment_ood_detection.py`.

Example:

```bash
./cli/run_experiment_calibration.sh --depth 4 --num_image 4000 --calib-args --no_figures
./cli/run_experiment_ood_detection.sh --num_nets 10 --ood-args --no_figures
```

Both wrappers print `EXP_ID` and `EXP_PATH` in their execution logs.

## Benchmark Experiments (CNN backbones)

Run benchmark training + evaluations:

```bash
./cli/run_benchmark_experiment.sh
```

Run benchmark evaluations only on an existing experiment:

```bash
./cli/run_benchmark_evaluation.sh <exp_id> [extra args]
```

Method interface:
- Calibration methods (`scripts/benchmark_calibration.py --methods`):
  `baseline,temp_scaling,vec_scaling,isotonic`
- OOD methods (`scripts/benchmark_ood_detection.py --methods`):
  `baseline,temp_scaling,odin,energy_score`

Benchmark outputs:
- `calibration_comparison.json`
- `ood_comparison.json`

## Finetuning Experiment

Run finetuning pretrained ResNet18 with/without warmup:

```bash
./cli/run_resnet_finetuning.sh
```

## Additional Experiments

| Command | Purpose | Output location |
| :--- | :--- | :--- |
| `./cli/run_initial_overconfidence.sh` | Initial overconfidence analysis (depth/output sweeps) | `exp/<exp_id>/` |
| `./cli/run_input_space.sh` | 2D input-space uncertainty analysis | `exp/<exp_id>/` |
| `./cli/run_sweep.sh` | Depth × num_image sweep with multi-net runs | `exp/<exp_id>/` + `summary.json` |

## Output Directory and Artifacts

Typical experiment directory:

```text
exp/<exp_id>/
├── config.json
├── training_results.json
├── calibration.json
├── ood_detection.json
├── checkpoints/
└── figures/
    ├── training/
    ├── calibration/
    ├── ood_detection/
    └── figure_paths.json
```

Artifact mapping:
- `experiment_training.py` writes `config.json`, `training_results.json`, and model files under `checkpoints/`.
- `experiment_calibration.py` writes `calibration.json` and calibration figures.
- `experiment_ood_detection.py` writes `ood_detection.json` and OOD figures.


## Corresponding Experiments and Figures

1. `cli/run_sweep.sh`: Fig. 1, Fig. 2f
2. `cli/run_experiment_calibration.sh`: Fig. 2d, Fig. 2e
3. `cli/run_benchmark_experiment.sh`: Fig. 2g
4. `cli/run_resnet_finetuning.sh`: Fig. 2h
5. `cli/run_input_space.sh`: Fig. 3a-d
6. `cli/run_initial_overconfidence.sh`: Fig. 3e-j
7. `cli/run_experiment_ood_detection.sh`: Fig. 4, Fig. 5

## Figure Regeneration from Source Data

`source_data/fig.*/` contains source JSON files and regeneration scripts.
Run the following from the repository root to regenerate all paper figures:

```bash
source .venv/bin/activate
for s in source_data/fig.*/script.py; do
  python "$s"
done
```

To regenerate only one figure group:

```bash
python source_data/fig.1/script.py
```

Each script overwrites/updates `*.svg` and `stats.json` in its own `source_data/fig.X/` directory.  
Statistical test outputs are also saved per figure group in that `stats.json` file.


### Citation

```bibtex
@article{cheon2026
  title={Brain-Inspired Warm-Up Training with Random Noise for Uncertainty Calibration},
  author={Cheon, Jeonghwan and Paik, Se-Bum},
  journal={Nature Machine Intelligence},
  year={2026}
}
```
