"""
Evaluation script for chord classification model.

Loads trained model and evaluates on test set.
Compatible with models trained using train.py

Usage:
    python src/models/evaluate.py
    python src/models/evaluate.py --model models/chord_classifier_model.pth
    python src/models/evaluate.py --model models/chord_classifier_model.pth --data data/processed/chord_dataset.pkl
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path

# Import model architecture
from chord_classifier import ChordClassifier


def load_checkpoint(model_path):
    """
    Load model checkpoint.

    Args:
        model_path: Path to .pth checkpoint file

    Returns:
        checkpoint dictionary
    """
    print(f"Loading model: {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, weights_only=False)

    # Display checkpoint info
    print(f"Input size: {checkpoint['input_size']}")
    print(f"Classes: {checkpoint['num_classes']} ({checkpoint['label_mapping']})")

    # Check for architecture
    if "architecture" in checkpoint:
        print(f"Architecture: {checkpoint['architecture']}")
    else:
        print("Architecture: [will be inferred from weights]")

    # Show training info if available
    if "dropout_rate" in checkpoint:
        print(f"Dropout: {checkpoint['dropout_rate']}")
    if "learning_rate" in checkpoint:
        print(f"Learning rate: {checkpoint['learning_rate']}")

    if "test_accuracy" in checkpoint:
        print(f"Recorded test accuracy: {checkpoint['test_accuracy']:.2f}%")

    return checkpoint


def create_model_from_checkpoint(checkpoint):
    """
    Recreate model from checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        Initialized model
    """
    # Get architecture
    if "architecture" in checkpoint:
        architecture = checkpoint["architecture"]
    else:
        # Try to infer from model state dict
        state_dict = checkpoint["model_state_dict"]
        architecture = []

        layer_idx = 1
        while f"hidden_layers.{layer_idx}.weight" in state_dict:
            architecture.append(
                state_dict[f"hidden_layers.{layer_idx}.weight"].shape[0]
            )
            layer_idx += 2  # Skip dropout layers

        print(f"Inferred architecture: {architecture}")

    # Get dropout rate
    if "dropout_rate" in checkpoint:
        dropout = checkpoint["dropout_rate"]
    elif "best_hyperparameters" in checkpoint:
        dropout = checkpoint["best_hyperparameters"].get("dropout", 0.15)
    else:
        dropout = 0.15
        print(f"Warning: Dropout not found, using default {dropout}")

    # Create model
    model = ChordClassifier(
        input_size=checkpoint["input_size"],
        hidden_sizes=architecture,
        num_classes=checkpoint["num_classes"],
        dropout_rate=dropout,
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set to evaluation mode

    print("Model loaded")

    return model


def load_test_data(dataset_path):
    """
    Load test data from preprocessed dataset.

    Args:
        dataset_path: Path to .pkl dataset file

    Returns:
        X_test, y_test, label_mapping
    """
    print(f"\nLoading test data: {dataset_path}")

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    X_test = data["X_test"]
    y_test = data["y_test"]
    label_mapping = data["label_mapping"]

    print(f"Test samples: {len(y_test)} ({X_test.shape[1]} features)")

    return X_test, y_test, label_mapping


def evaluate_model(model, X_test, y_test, label_mapping, device="cpu", batch_size=32):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_mapping: Dictionary mapping class names to indices
        device: Device to run evaluation on
        batch_size: Batch size for evaluation

    Returns:
        accuracy, predictions, labels, confusion_matrix
    """
    model = model.to(device)
    model.eval()

    # Convert to tensors and create DataLoader
    X_test_tensor = torch.FloatTensor(X_test)
    test_dataset = TensorDataset(X_test_tensor, torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Collect predictions
    all_predictions = []
    all_labels = []

    print("\nEvaluating...")
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = 100 * np.sum(all_predictions == all_labels) / len(all_labels)
    cm = confusion_matrix(all_labels, all_predictions)

    return accuracy, all_predictions, all_labels, cm


def print_evaluation_results(accuracy, predictions, labels, cm, label_mapping):
    """
    Print detailed evaluation results.

    Args:
        accuracy: Overall accuracy
        predictions: Model predictions
        labels: True labels
        cm: Confusion matrix
        label_mapping: Label to index mapping
    """
    print("\n" + "=" * 70)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(
        f"Correct: {np.sum(predictions == labels)}/{len(labels)}, Incorrect: {np.sum(predictions != labels)}"
    )
    print("=" * 70)

    # Create reverse mapping (index -> label)
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}
    target_names = [reverse_mapping[i] for i in range(len(label_mapping))]

    # Per-class statistics
    print("\nPer-class results:")
    for class_idx, class_name in enumerate(target_names):
        class_mask = labels == class_idx
        class_correct = np.sum((predictions == labels) & class_mask)
        class_total = np.sum(class_mask)
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0

        print(f"\n{class_name}: {class_acc:.2f}% ({class_correct}/{class_total})")

        # Show confusion with other classes
        if class_total > 0:
            class_errors = predictions[class_mask]
            for other_idx, other_name in enumerate(target_names):
                if other_idx != class_idx:
                    n_confused = np.sum(class_errors == other_idx)
                    if n_confused > 0:
                        pct = 100 * n_confused / class_total
                        print(
                            f"  -> confused with {other_name}: {n_confused} ({pct:.1f}%)"
                        )

    # Classification report
    print("\nDetailed metrics:")
    print(
        classification_report(labels, predictions, target_names=target_names, digits=3)
    )

    # Confusion matrix
    print("Confusion matrix:")
    print(f"{'':12}", end="")
    for name in target_names:
        print(f"{name:>10}", end="")
    print()

    for i, name in enumerate(target_names):
        print(f"{name:12}", end="")
        for j in range(len(target_names)):
            print(f"{cm[i, j]:>10}", end="")
        print()


def plot_confusion_matrix(cm, label_mapping, save_path="confusion_matrix.png"):
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix
        label_mapping: Label to index mapping
        save_path: Path to save figure
    """
    # Create reverse mapping
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}
    class_names = [reverse_mapping[i] for i in range(len(label_mapping))]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix",
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    return fig


def save_results(
    accuracy, predictions, labels, cm, save_path="results/evaluation_results.txt"
):
    """
    Save evaluation results to text file.

    Args:
        accuracy: Overall accuracy
        predictions: Model predictions
        labels: True labels
        cm: Confusion matrix
        save_path: Path to save results
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        f.write("Chord Classification - Test Evaluation\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Correct: {np.sum(predictions == labels)}/{len(labels)}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    print(f"Results saved: {save_path}")


def main():
    """Main evaluation pipeline."""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate chord classification model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/chord_classifier_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/chord_dataset.pkl",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on (cpu/cuda)",
    )
    parser.add_argument(
        "--save-results", action="store_true", help="Save results to file"
    )
    parser.add_argument(
        "--plot-cm", action="store_true", help="Plot and save confusion matrix"
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 70)
    print("Chord Classification - Model Evaluation")
    print("=" * 70 + "\n")

    # Load checkpoint
    checkpoint = load_checkpoint(args.model)

    model = create_model_from_checkpoint(checkpoint)

    X_test, y_test, label_mapping = load_test_data(args.data)

    if checkpoint["label_mapping"] != label_mapping:
        print("\nWarning: Label mapping mismatch!")
        print(f"  Model: {checkpoint['label_mapping']}")
        print(f"  Data:  {label_mapping}")

    accuracy, predictions, labels, cm = evaluate_model(
        model,
        X_test,
        y_test,
        label_mapping,
        device=args.device,
        batch_size=args.batch_size,
    )

    print_evaluation_results(accuracy, predictions, labels, cm, label_mapping)

    if args.save_results:
        save_results(accuracy, predictions, labels, cm)

    if args.plot_cm:
        plot_confusion_matrix(cm, label_mapping)

    if "test_accuracy" in checkpoint:
        recorded = checkpoint["test_accuracy"]
        diff = accuracy - recorded
        print(f"\nRecorded accuracy: {recorded:.2f}%")
        if abs(diff) < 0.01:
            print(f"Match! (diff: {diff:+.2f}%)")
        else:
            print(f"Difference: {diff:+.2f}%")

    print("\n" + "=" * 70 + "\n")

    return accuracy


if __name__ == "__main__":
    main()
