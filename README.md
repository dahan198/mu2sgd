# Do Stochastic, Feel Noiseless: Stable Stochastic Optimization via a Double Momentum Mechanism
[![ICLR 2025](https://img.shields.io/badge/ICLR-2025-blue.svg)](https://openreview.net/pdf?id=zCZnEXF3bN) [![arXiv](https://img.shields.io/badge/arXiv-2304.04169-B31B1B.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Official repository for the paper **"Do Stochastic, Feel Noiseless: Stable Stochastic Optimization via a Double Momentum Mechanism"** by **Tehila Dahan, Kfir Y. Levy**, accepted to **ICLR 2025**.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/dahan198/mu2sgd.git
cd mu2sgd
```

### 2. Install Dependencies

First, ensure that PyTorch is installed. You can install it by selecting the appropriate command based on your environment from [PyTorch's official website](https://pytorch.org/get-started/locally/).

#### Install Other Dependencies

After installing PyTorch, install the remaining dependencies using:

```bash
pip install -r requirements.txt
```

---

## Running Experiments
Convex Setting - MNIST
```bash
python run_mnist_convex.py
```

Non-Convex Setting - MNIST
```bash
python run_mnist_non_convex.py
```

Non-Convex Setting - CIFAR-10
```bash
python run_cifar10_non_convex.py
```

## Logging

- **Weights & Biases (wandb)**: This repository supports logging with [Weights & Biases](https://wandb.ai/) for experiment tracking and visualization. Ensure you have a `wandb.yaml` file in the `config` directory with your project and entity name.
  
  Example `wandb.yaml`:
  ```yaml
  project: "mu2sgd experiments"
  entity: "your-wandb-username"
  ```

## Citation
If you find this repository useful, please cite:

```bibtex
@inproceedings{mu2sgd2025,
  title={Do Stochastic, Feel Noiseless: Stable Stochastic Optimization via a Double Momentum Mechanism},
  author={Tehila Dahan and Kfir Y. Levy},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.