"""
Neural Network classifier for chord classification.

Architecture:
    Input(48) → Dense(64) → ReLU → Dropout(0.3) →
    Dense(32) → ReLU → Dropout(0.3) →
    Dense(16) → ReLU →
    Dense(3) → Softmax
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChordClassifier(nn.Module):
    """
    Multi-layer Perceptron for chord classification.

    Args:
        input_size: Number of input features (48 for one-hot encoding)
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes (3 for Major/minor/Other)
        dropout_rate: Dropout probability for regularization
    """

    def __init__(
        self,
        input_size: int = 48,
        hidden_sizes: list = [64, 32, 16],
        num_classes: int = 3,
        dropout_rate: float = 0.3,
    ):
        super(ChordClassifier, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Remove last dropout
        layers = layers[:-1]

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        # Combine all layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)

    def predict(self, x):
        """
        Make predictions (returns class indices).

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, x):
        """
        Get class probabilities.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


# Simple baseline model (smaller, faster)
class SimpleChordClassifier(nn.Module):
    """
    Simpler baseline model with fewer parameters.

    Architecture: Input(48) → Dense(32) → ReLU → Dense(3) → Softmax
    """

    def __init__(self, input_size: int = 48, num_classes: int = 3):
        super(SimpleChordClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
