# DAM Optimizer

DAM is a novel optimization algorithm designed for deep learning models. This repository contains the implementation of DAM along with benchmark comparisons against the Adan optimizer.

## Repository Structure

- `DAM.py`: Implementation of the DAM optimizer
- `benchmark.py`: Scripts for benchmarking DAM against Adan
- `train.ipynb`: Jupyter notebook with training examples
- `environment.yml`: Conda environment configuration file

## Key Features

- Competitive performance compared to Adan optimizer
- Memory-efficient implementation
- Easy integration with PyTorch models

## Benchmark Results

(Include your benchmark results comparing DAM vs Adan here. You may want to add:
- Convergence speed comparison
- Final accuracy metrics
- Memory usage statistics
- Training time comparisons)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/22w1006/DAM-Opt-project
   ```

2. Set up the environment:
   ```bash
   conda env create -f environment.yml
   conda activate dam_env
   ```

## Usage

Basic usage in PyTorch:
```python
from DAM import DAM

optimizer = DAM(model.parameters(), lr=0.001)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

