# ADGAT: Adversarial Defense Graph Attention Network

This repository contains the implementation of ADGAT (Adversarial Defense Graph Attention Network), a research project focused on defending against adversarial attacks on graph neural networks.

## Project Structure

```
ADGAT_Code/
├── ADGAT/                  # Main implementation
│   ├── layer.py           # Graph attention layer implementation
│   ├── model.py           # ADGAT model and anomaly detector
│   ├── train.py           # Training script
│   ├── utils.py           # Utility functions
│   ├── run.sh             # Shell script to run experiments
│   └── checkpoint/        # Model checkpoints
├── data/                  # Dataset files
│   ├── cora/              # Cora dataset and attack variants
│   ├── citeseer/          # CiteSeer dataset and attack variants
│   └── cora_ml/           # Cora-ML dataset and attack variants
└── params/                # Trained detector parameters
```

## Datasets

The project supports three graph datasets:
- **Cora**: Scientific publications dataset
- **CiteSeer**: Scientific publications dataset
- **Cora-ML**: Machine learning papers subset

Each dataset includes variants with different types of adversarial attacks:
- **Meta Attack**: Various perturbation rates (0.05, 0.1, 0.15, 0.2, 0.25)
- **Nettack**: Different attack intensities (1.0, 2.0, 3.0, 4.0, 5.0)
- **H-Attack**: Various perturbation rates (0.05, 0.1, 0.15, 0.2, 0.25)

## Usage

### Training

Run the training script with different attack scenarios:

```bash
# Meta attack on Cora
python3 ADGAT/train.py --dataset cora --th1 0.75 --th2 0.9 --ptb 0.05 --attack meta

# Meta attack on CiteSeer
python3 ADGAT/train.py --dataset citeseer --th1 0.7 --th2 0.85 --ptb 0.05 --attack meta

# Meta attack on Cora-ML
python3 ADGAT/train.py --dataset cora_ml --th1 0.8 --th2 0.95 --ptb 0.05 --attack meta

# Nettack on Cora
python3 ADGAT/train.py --dataset cora --th1 0.72 --th2 0.8 --ptb 0.05 --attack nettack

# H-attack on CiteSeer
python3 ADGAT/train.py --dataset citeseer --th1 0.7 --th2 0.85 --ptb 0.05 --attack hattack
```

### Parameters

- `--dataset`: Choose from {cora, cora_ml, citeseer, polblogs}
- `--attack`: Attack type {meta, nettack, hattack}
- `--th1`, `--th2`: Thresholds for link prediction
- `--ptb`: Perturbation rate
- `--epochs`: Number of training epochs (default: 10000)
- `--lr`: Learning rate (default: 5e-3)
- `--hidden`: Number of hidden units (default: 8)
- `--nb_heads`: Number of attention heads (default: 8)

## Model Architecture

The ADGAT model consists of:
1. **Graph Attention Layers**: Multi-head attention mechanism for node feature aggregation
2. **Anomaly Detector**: Neural network for detecting adversarial perturbations
3. **Link Prediction**: Component for predicting legitimate graph connections

## Requirements

- Python
- PyTorch
- NumPy
- SciPy

## Files Description

- `model.py`: Contains the main ADGAT model and AnomalyDetector classes
- `layer.py`: Implements the graph attention layer
- `train.py`: Main training script with argument parsing
- `utils.py`: Utility functions for data processing and evaluation
- `run.sh`: Batch script for running multiple experiments

## Checkpoints

Pre-trained model checkpoints are available in the `checkpoint/` directory:
- `ckpt_cora_best.pth`
- `ckpt_citeseer_best.pth`
- `ckpt_cora_ml_best.pth`

