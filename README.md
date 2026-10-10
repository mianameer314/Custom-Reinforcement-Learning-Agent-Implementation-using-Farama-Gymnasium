# Custom Reinforcement Learning Agent Implementation using Farama Gymnasium

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0%2B-brightgreen.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview
This repository implements a modular Reinforcement Learning (RL) framework from scratch using NumPy and Farama Gymnasium's `FrozenLake-v1` benchmark environment. It includes:
- **Tabular Q-Learning** (Off-Policy Temporal Difference Control with Bellman Optimality).
- **SARSA** (On-Policy TD Control Baseline).
- **Automated Experimentation & Hyperparameter Tuning** ($\alpha$, $\gamma$, $\epsilon$-decay).
- **Visual Policy Rollout Rendering** to animated GIF.
- **Formal 4-Page Technical Report (PDF)** adhering to COMSATS academic standards.

### 🎯 Key Performance Highlights
- **Evaluation Success Rate (100 Deterministic Episodes):** `100.00%`
- **Mean Episodic Reward:** `+1.00`
- **Optimal Step Trajectory:** Exactly `6.00 Steps` ($S \to F(4) \to F(8) \to F(9) \to F(10) \to F(14) \to G(15)$)
- **Catastrophic Failure Rate:** `0.00%` (Zero hole pitfalls)

---

## 📂 Modular Repository Architecture

```text
├── src/
│   ├── __init__.py                     # Package interface exports
│   ├── agent.py                        # Modular Q-Learning & SARSA agents from scratch
│   └── utils.py                        # Plotting, evaluation & custom grid visualizer
├── notebooks/
│   └── reinforcement_learning_assignment.ipynb # Executed Jupyter Notebook with training curves
├── results/
│   ├── q_table.npy                     # Learned optimal Q-table weights (16x4)
│   ├── trained_agent.gif               # Animated visual policy rollout
│   ├── training_curves.png             # Episodic reward, step count, and TD error curves
│   └── comparison_curve.png            # Q-Learning vs SARSA & learning rate sensitivity
├── report/
│   ├── Technical_Report.pdf            # 4-Page publication-grade PDF repor
│   └── comsats_logo.png                # Official university emblem
├── main.py                             # Entry point for training, evaluation & GIF generation
├── requirements.txt                    # Project dependency specification
├── .gitignore                          # Standard Python & IDE ignore rules
└── README.md                           # Project documentation
```

---

## 🚀 Quickstart & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train & Evaluate the Agent
Run the end-to-end training and evaluation pipeline:
```bash
python main.py --env FrozenLake-v1 --episodes 2000 --save_dir results
```

---

## 📊 Experimental Results & Policy

| Evaluation Metric | Measured Value | Benchmark / Target |
| :--- | :---: | :---: |
| **Deterministic Success Rate (100 Ep)** | **100.00%** | 100.0% (Flawless Navigation) |
| **Mean Reward per Episode** | **+1.00** | +1.00 (Max Possible) |
| **Mean Steps to Goal** | **6.00 Steps** | 6 Steps (Theoretical Minimum) |
| **Hole Pitfall Rate** | **0.00%** | 0.0% (Zero Catastrophic Failures) |

### 🤖 Learned Optimal Policy Grid $\pi^*(s)$
```text
+-------+-------+-------+-------+
| START | RIGHT | DOWN  | LEFT  |  (Row 0: States 0-3)
+-------+-------+-------+-------+
| DOWN  | HOLE  | DOWN  | HOLE  |  (Row 1: States 4-7)
+-------+-------+-------+-------+
| RIGHT | DOWN  | DOWN  | HOLE  |  (Row 2: States 8-11)
+-------+-------+-------+-------+
| HOLE  | RIGHT | RIGHT | GOAL  |  (Row 3: States 12-15)
+-------+-------+-------+-------+
```

---

## 📜 Deliverables Checklist
- [x] **Modular Python Source Code:** `src/agent.py`, `src/utils.py`, `main.py`, `requirements.txt`
- [x] **Executed Jupyter Notebook:** `notebooks/reinforcement_learning_assignment.ipynb`
- [x] **Saved Model Weights:** `results/q_table.npy`
- [x] **Rollout Animation GIF:** `results/trained_agent.gif`
- [x] **4-Page Technical Report (PDF):** `report/Technical_Report.pdf`
- [x] **Clean .gitignore & Git Repository Ready**
