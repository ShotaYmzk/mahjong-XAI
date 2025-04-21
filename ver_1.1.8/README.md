# Advanced Transformer-based Mahjong AI

A state-of-the-art Mahjong AI that uses Transformer neural networks to make optimal discard decisions in Japanese Riichi Mahjong. This project implements an advanced deep learning model that learns from professional player data to develop strategic gameplay.

## Overview

This Mahjong AI system uses modern Transformer architecture with self-attention mechanisms to understand the complex sequential nature of Mahjong games. It processes the game history, current hand state, and other players' actions to predict the optimal tile to discard at each decision point.

### Key Features

- **Transformer Neural Network**: Uses an 8-layer Transformer encoder with rotary positional encoding for state-of-the-art sequence modeling
- **Imitation Learning**: Trained on data from high-level Tenhou.net players to mimic expert gameplay
- **Data Augmentation**: Expands training data through intelligent augmentation techniques
- **Game State Tracking**: Complete implementation of Mahjong rules and game state management
- **Shanten Calculation**: Evaluates hand efficiency using shanten (distance to ready) algorithm
- **Detailed Decision Explanations**: Provides reasoning for each discard decision

## Project Structure

- `model.py`: Data processing and batch generation for model training
- `train.py`: Advanced Transformer model definition and training pipeline
- `mahjong_ai.py`: Inference engine for running the trained model
- `game_state.py`: Complete implementation of Mahjong game rules and state tracking
- `feature_extractor_imitation.py`: Feature extraction for the model input
- `tile_utils.py`: Utility functions for tile representation and conversion
- `naki_utils.py`: Functions for handling calls (chi, pon, kan)
- `shanten.py`: Implementation of shanten calculation algorithm

## Transformer Architecture

The AI uses a specialized Transformer architecture with:

- **Rotary Positional Encoding**: Instead of standard sinusoidal, uses RoPE for better sequence understanding
- **Multi-head Self-attention**: 12 attention heads to capture complex relationships in game events
- **Pre-norm Architecture**: More stable training with layer normalization before attention
- **GELU Activation**: Improved activation function compared to ReLU
- **Attention Pooling**: Learns to focus on the most important events in the game history

## Training Process

The model is trained through imitation learning on high-quality game logs:

1. **Data Collection**: Parse Tenhou.net XML game logs from professional and high-ranked players
2. **Feature Extraction**: Convert raw game logs into numeric features capturing game state
3. **Data Augmentation**: Increase dataset size through intelligent variations of training examples
4. **Model Training**: Train using advanced techniques (mixed precision, EMA, cosine scheduling)
5. **Evaluation**: Measure performance using accuracy and expert-based metrics

## Usage

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy, tqdm, matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mahjong-transformer-ai.git
cd mahjong-transformer-ai

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Place your Tenhou.net XML game logs in the `xml_logs/` directory
2. Run the data processing script:

```bash
python model.py
```

### Training

Train the model with:

```bash
python train.py
```

This will:
- Load the processed data
- Train the Transformer model
- Save checkpoints and the best model
- Generate training curves

### Making Decisions

Use the trained model for discard decisions:

```bash
python mahjong_ai.py
```

## Performance

The model achieves:

- **Top-1 Accuracy**: 65-70% match with professional player decisions
- **Top-3 Accuracy**: 85-90% match with professional player decisions
- **Strategic Consistency**: Strong tendency toward efficient hand-building

## Future Work

- **Reinforcement Learning**: Extend beyond imitation to learn optimal strategies through self-play
- **Policy Distillation**: Create smaller, faster models for deployment
- **Opponent Modeling**: Adapt strategy based on opponent playing styles
- **End-to-End Game Play**: Complete AI system that can play full games autonomously

## Acknowledgments

- Implementation inspired by Transformer architectures from "Attention is All You Need" (Vaswani et al.)
- Mahjong-specific algorithms adapted from various open-source projects
- Trained using data from Tenhou.net

## License

This project is licensed under the MIT License - see the LICENSE file for details. 