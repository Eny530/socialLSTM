# Social LSTM

A PyTorch implementation of the Social LSTM model for human trajectory prediction, based on the paper ["Social LSTM: Human Trajectory Prediction in Crowded Spaces"](https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf) by Alahi et al.

## Overview

This implementation provides a clean, modular, and optimized codebase for the Social LSTM model. The project is structured to be easily understandable and extensible, allowing for experimentation with different model architectures and datasets.

The Social LSTM model accounts for the interactions between pedestrians in a scene, enabling more accurate predictions of future trajectories. This is achieved through a social pooling mechanism that captures the spatial configuration of neighboring pedestrians.

## Features

- **Social LSTM Model**: Implementation of the Social LSTM model with social pooling mechanism.
- **Vanilla LSTM Baseline**: A simple LSTM baseline for comparison.
- **ETH-UCY Dataset Support**: Support for the ETH-UCY pedestrian trajectory dataset.
- **Visualization Tools**: Tools for visualizing trajectories and model performance.
- **Metrics**: Implementation of standard metrics for trajectory prediction (ADE, FDE).
- **Data Augmentation**: Random rotation for data augmentation.

## Project Structure

```
social-lstm/
├── config.py                 # Configuration parameters
├── data/                     # Data directory
│   └── README.md             # Data information
├── datasets/                 # Dataset loaders
│   ├── __init__.py
│   ├── dataset.py            # Base dataset class
│   └── eth_ucy.py            # ETH-UCY dataset loader
├── models/                   # Model implementations
│   ├── __init__.py
│   ├── social_lstm.py        # Social LSTM model
│   └── vanilla_lstm.py       # Vanilla LSTM baseline
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── data_utils.py         # Data processing utilities
│   ├── losses.py             # Loss functions
│   ├── metrics.py            # Evaluation metrics
│   └── visualization.py      # Visualization tools
├── train.py                  # Training script
├── eval.py                   # Evaluation script
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/social-lstm.git
   cd social-lstm
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the ETH-UCY dataset:
   ```bash
   # Download script will be provided
   ```

## Usage

### Training

To train the Social LSTM model on the ETH dataset:

```bash
python train.py --model social_lstm --dataset eth --num_epochs 100
```

For the Vanilla LSTM baseline:

```bash
python train.py --model vanilla_lstm --dataset eth --num_epochs 100
```

### Evaluation

To evaluate a trained model:

```bash
python eval.py --model social_lstm --dataset eth --checkpoint_path ./checkpoints/social_lstm_eth_20230101_000000/best_model.pt
```

### Configuration

The model and training parameters can be configured in `config.py`. Key parameters include:

- `--model`: Model type (social_lstm or vanilla_lstm)
- `--dataset`: Dataset name (eth, hotel, univ, zara1, zara2)
- `--obs_len`: Observed trajectory length
- `--pred_len`: Prediction trajectory length
- `--embedding_dim`: Embedding dimension
- `--hidden_dim`: Hidden state dimension
- `--num_layers`: Number of LSTM layers
- `--dropout`: Dropout probability
- `--neighborhood_size`: Neighborhood size for social pooling (in meters)
- `--grid_size`: Grid size for social pooling
- `--batch_size`: Batch size
- `--num_epochs`: Number of epochs
- `--learning_rate`: Learning rate

## Results

Performance on the ETH-UCY dataset (ADE/FDE in meters):

| Method       | ETH    | HOTEL  | UNIV   | ZARA1  | ZARA2  | AVG    |
|--------------|--------|--------|--------|--------|--------|--------|
| Vanilla LSTM | -/-    | -/-    | -/-    | -/-    | -/-    | -/-    |
| Social LSTM  | -/-    | -/-    | -/-    | -/-    | -/-    | -/-    |

(Fill in with your results after training)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Social LSTM Paper](https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf) by Alahi et al.
- ETH-UCY Dataset by Pellegrini et al. and Lerner et al.
