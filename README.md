# Chord Classification with Uncertainty Estimation

A deep learning approach to musical chord classification using neural networks with Monte Carlo Dropout for uncertainty quantification. This project classifies MIDI chords into three categories: Major, minor, and Other, while providing confidence estimates and uncertainty decomposition.

## Group members
- Riandika Marshequila Dinu
- Yitong Lin Yang
- Dariana Hosu

## Project Overview

This project implements a neural network classifier for musical chords with the following key features:

- **Multi-class classification**: Major, minor, and Other chord types
- **Uncertainty quantification**: MC Dropout for aleatoric and epistemic uncertainty estimation
- **High accuracy**: Achieves ~80%+ test accuracy, significantly outperforming baseline models
- **Interpretable predictions**: Provides confidence scores and uncertainty breakdowns for each prediction

## Features

### Core Functionality
- **Neural Network Architecture**: Fully connected network with dropout regularization
  - Input: 48 features (MIDI note representations)
  - Hidden layers: [64, 32, 16]
  - Output: 3 classes (Major, minor, Other)
  - Total parameters: ~5,800

- **Monte Carlo Dropout**: Uncertainty estimation through stochastic forward passes
  - Decomposes total uncertainty into:
    - **Aleatoric uncertainty**: Data/irreducible uncertainty (~80% of total)
    - **Epistemic uncertainty**: Model/knowledge uncertainty (~20% of total)
  - Provides confidence scores and prediction consistency metrics

- **Training Features**:
  - Early stopping with validation monitoring
  - 60/20/20 train/validation/test split
  - Adam optimizer with learning rate 0.001
  - Dropout rate: 0.1

### Visualization
- Training history curves (loss and accuracy)
- Comprehensive uncertainty analysis (6-panel visualization)
- Per-class performance metrics
- Confusion matrices
- Model comparison figures

## Results Summary

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 81.47% |
| **Test Accuracy (MC Dropout)** | 80.93% |
| **Training Accuracy** | 85.34% |
| **Validation Accuracy** | 82.98% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Major** | 0.84 | 0.87 | 0.85 | 1823 |
| **Other** | 0.72 | 0.62 | 0.67 | 498 |
| **minor** | 0.81 | 0.81 | 0.81 | 1333 |

### Uncertainty Analysis

- **Mean Confidence**: 0.8070 ± 0.1672
- **Total Uncertainty**: 0.4475 ± 0.2723
- **Aleatoric Uncertainty**: 0.4070 (data noise)
- **Epistemic Uncertainty**: 0.0847 (model uncertainty)

**Key Insight**: Low epistemic uncertainty (20%) indicates the model has learned well. High aleatoric uncertainty (80%) suggests remaining errors are due to inherent data ambiguity rather than insufficient training.

### Model Comparison

| Model | Test Accuracy | Parameters |
|-------|---------------|------------|
| Random (Weighted) | 39.0% | - |
| Majority Class | 47.6% | - |
| Logistic Regression | 50.3% | 147 |
| **Neural Network** | **81.5%** | **5,795** |

## Getting Started

### Prerequisites

```bash
# Python 3.8+
pip install torch numpy scikit-learn matplotlib seaborn
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/grp2-quartet-chord-classification.git
cd grp2-quartet-chord-classification

# Install dependencies
pip install -r requirements.txt
```


## Usage

### 1. Training the Model

Train a new model from scratch:

```bash
python src/models/train.py
```

The training script will:
- Load the preprocessed dataset
- Train with early stopping (patience=15)
- Evaluate on test set
- Run MC Dropout uncertainty estimation (100 samples)
- Save model checkpoint and results
- Generate training curves and uncertainty plots

**Output files:**
- `chord_classifier_model.pth` - Model checkpoint with metadata
- `uncertainty_results.pkl` - Detailed uncertainty results
- `training_history.png` - Loss and accuracy curves
- `uncertainty_analysis.png` - 6-panel uncertainty visualization

### 2. Evaluating the Model

Evaluate a trained model:

```bash
# Basic evaluation
python src/models/evaluate.py

# With options
python src/models/evaluate.py \
    --model chord_classifier_model.pth \
    --data data/processed/chord_dataset.pkl \
    --plot-cm \
    --save-results
```

**Arguments:**
- `--model`: Path to model checkpoint (default: `chord_classifier_model.pth`)
- `--data`: Path to dataset (default: `data/processed/chord_dataset.pkl`)
- `--batch-size`: Batch size for evaluation (default: 32)
- `--device`: Device (cpu/cuda) (default: cpu)
- `--save-results`: Save results to file
- `--plot-cm`: Plot confusion matrix

### 3. Generating Figures

Generate publication-quality figures:

```bash
python src/models/figures.py \
    --model chord_classifier_model.pth \
    --data data/processed/chord_dataset.pkl \
    --output-dir figures
```

This generates:
- Per-class performance analysis
- Model comparison figures
- Training history plots


## Technical Details

### Model Architecture

```python
ChordClassifier(
    input_size=48,           # MIDI note features
    hidden_sizes=[64, 32, 16],
    num_classes=3,           # Major, minor, Other
    dropout_rate=0.1
)
```

**Architecture:**
- Input Layer: 48 features
- Hidden Layer 1: 64 neurons + ReLU + Dropout(0.1)
- Hidden Layer 2: 32 neurons + ReLU + Dropout(0.1)
- Hidden Layer 3: 16 neurons + ReLU + Dropout(0.1)
- Output Layer: 3 classes (softmax)

### Training Configuration

```python
{
    'learning_rate': 0.001,
    'dropout_rate': 0.1,
    'batch_size': 32,
    'max_epochs': 100,
    'patience': 15,           # Early stopping
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss'
}
```

### Dataset Information

- **Total samples**: 18,267
  - Training: 10,959 (60%)
  - Validation: 3,654 (20%)
  - Test: 3,654 (20%)

- **Class distribution** (test set):
  - Major: 1,823 samples (49.9%)
  - minor: 1,333 samples (36.5%)
  - Other: 498 samples (13.6%)

- **Features**: 48-dimensional vectors representing MIDI note information

### Uncertainty Quantification

**MC Dropout** performs multiple stochastic forward passes with dropout enabled:

```python
# Total Uncertainty = Aleatoric + Epistemic
Predictive Entropy = Expected Entropy + Mutual Information
                     (aleatoric)       (epistemic)
```

**Interpretation:**
- **High aleatoric, low epistemic** → Model learned well, data is noisy
- **Low aleatoric, high epistemic** → Need more training data
- **High both** → Difficult examples, need more data and/or better features

## Key Findings

### 1. Model Performance
- Neural network achieves **81.5% accuracy**, a **31.2% improvement** over logistic regression baseline (50.3%)
- Excellent performance on Major (87% recall) and minor (81% recall) chords
- "Other" class is challenging (62% recall) due to diverse chord types

### 2. Uncertainty Calibration
- **Well-calibrated**: Incorrect predictions have ~1.7x higher uncertainty than correct ones
- Correct predictions: 0.84 confidence, 0.40 uncertainty
- Incorrect predictions: 0.66 confidence, 0.67 uncertainty

### 3. Uncertainty Decomposition
- **Aleatoric dominance** (~83%) indicates model has extracted most learnable patterns
- **Low epistemic** (~17%) means adding more training data would have diminishing returns
- Model knows what it knows and what it doesn't know

### 4. Per-Class Insights
- **Other class** has highest uncertainty (both aleatoric and epistemic)
- This suggests these chords are inherently more ambiguous
- May benefit from splitting into sub-categories (diminished, augmented, sus, etc.)

## Future Work

### Short-term Improvements
- [ ] Implement class weighting to improve "Other" class performance
- [ ] Add more sophisticated audio features (spectral, harmonic)
- [ ] Experiment with different architectures (deeper networks, residual connections)
- [ ] Add L2 regularization to reduce overfitting

### Medium-term Goals
- [ ] Split "Other" category into specific chord types (dim, aug, sus, 7th, etc.)
- [ ] Implement ensemble methods for improved accuracy
- [ ] Create real-time inference pipeline for MIDI input

### Long-term Vision
- [ ] Extend to full chord progressions and harmonic analysis
- [ ] Add temporal modeling (RNN/LSTM) for chord sequences
- [ ] Multi-task learning (chord + key + inversion detection)


## References

- **MC Dropout**: Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation"
- **Uncertainty Decomposition**: Depeweg et al. (2018). "Decomposition of Uncertainty"
- **Chord Recognition**: Humphrey & Bello (2012). "From Music Audio to Chord Tablature"
