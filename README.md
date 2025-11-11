# PhysiQ: Off-site Quality Assessment of Exercise in Physical Therapy

[![Paper](https://img.shields.io/badge/arXiv-2211.08245-b31b1b.svg)](https://arxiv.org/abs/2211.08245)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3570349-blue)](https://doi.org/10.1145/3570349)

**IMWUT/UbiComp 2023**
Hanchen David Wang, Meiyi Ma - Vanderbilt University

## Overview

PhysiQ is a smartwatch-based framework for quantitative assessment of physical therapy exercises. It uses a multi-task spatio-temporal Siamese Neural Network to measure exercise quality through classification (89.67% accuracy) and similarity comparison (R²=0.949).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run classification evaluation
python main.py --classification --exercise e1 --metric rom

# Run similarity comparison
python main.py --similarity --exercise e1 --metric rom

# Compare with baselines
python main.py --baseline --exercise e1 --metric rom
```

## Repository Structure

```
PhysiQ_IMWUT/
├── main.py                     # Main evaluation script
├── config.py                   # Configuration
├── models/                     # Neural network models
├── dataset_utils/              # Data loading
├── utils/                      # Utilities
├── preprocessing/              # Data preprocessing
└── scripts/                    # Analysis scripts
```

## Usage

**Tasks:**

- `--classification` - Evaluate classification performance (LOOCV)
- `--similarity` - Evaluate similarity comparison
- `--baseline` - Compare with baseline models
- `--full` - Run complete evaluation

**Exercises:**

- `e1` - Shoulder Abduction (default)
- `e2` - External Rotation
- `e3` - Forward Flexion

**Metrics:**

- `rom` - Range of Motion (default)
- `stability` - Movement stability
- `repetition` - Repetition counting

## Model Architecture

```
Input (6-channel IMU) → Sliding Windows → CNN → LSTM → Attention
                                                      ↓
                                          ┌───────────┴───────────┐
                                          ↓                       ↓
                                  Classification            Siamese
```

## Results

**Classification:** 89.67% accuracy (Shoulder Abduction ROM)
**Similarity:** R²=0.949 (intra-subject correlation)
**Dataset:** 31 participants, 1,550+ repetitions

## Citation

```bibtex
@article{wang2022physiq,
  title={PhysiQ: Off-site Quality Assessment of Exercise in Physical Therapy},
  author={Wang, Hanchen David and Ma, Meiyi},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={6},
  number={4},
  pages={1--31},
  year={2022},
  publisher={ACM},
  doi={10.1145/3570349}
}
```

## Links

- [Paper (arXiv)](https://arxiv.org/abs/2211.08245)
- [ACM Digital Library](https://doi.org/10.1145/3570349)
- [iOS App](https://github.com/HCWDavid/PhysiQApp)

## Contact

**Hanchen David Wang** - hanchen.wang.1@vanderbilt.edu

---

**Note:** Dataset not publicly available due to IRB/privacy restrictions.
**License:** Academic and research use only.
