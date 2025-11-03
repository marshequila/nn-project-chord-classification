"""
Logistic Regression Baseline for Chord Classification

This serves as a baseline to compare against the neural network.
Logistic Regression is a standard baseline for multi-class classification tasks.
"""

import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time


def load_dataset(dataset_path: str):
    """Load the preprocessed dataset."""
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded dataset from {dataset_path}")
    print(f"  Training samples: {len(data['X_train'])}")
    print(f"  Validation samples: {len(data['X_val'])}")
    print(f"  Test samples: {len(data['X_test'])}")
    print(f"  Input features: {data['input_size']}")
    print(f"  Number of classes: {data['num_classes']}")
    print(f"  Classes: {data['label_mapping']}")

    return data


def train_logistic_regression(X_train, y_train, max_iter=1000, random_state=42):
    """
    Train Logistic Regression classifier.

    Args:
        X_train: Training features (N, 48)
        y_train: Training labels (N,)
        max_iter: Maximum iterations for solver
        random_state: Random seed for reproducibility

    Returns:
        Trained LogisticRegression model
    """
    print("\nTraining Logistic Regression...")
    print("-" * 70)

    # Create model with L2 regularization (default)
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",  #solver (multinomial)
        verbose=0,
    )

    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f} seconds")
    print(f"  Solver: {model.solver}")
    print(f"  Multi-class strategy: {model.multi_class}")
    print(f"  Regularization (C): {model.C}")
    print(f"  Number of iterations: {model.n_iter_}")

    return model


def evaluate_model(model, X, y, label_mapping, dataset_name="Test"):
    """
    Evaluate model and print detailed metrics.

    Args:
        model: Trained model
        X: Features
        y: True labels
        label_mapping: Dictionary mapping class names to indices
        dataset_name: Name of dataset being evaluated
    """
    print(f"\n{dataset_name} Set Evaluation")
    print("-" * 70)

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Create reverse mapping (index -> label)
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}
    target_names = [reverse_mapping[i] for i in range(len(label_mapping))]

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(f"{'':10} {' '.join([f'{name:10}' for name in target_names])}")
    for i, row in enumerate(cm):
        print(f"{target_names[i]:10} {' '.join([f'{val:10}' for val in row])}")

    return accuracy


def main():
    """Main baseline evaluation pipeline."""

    print("=" * 70)
    print("LOGISTIC REGRESSION BASELINE - CHORD CLASSIFICATION")
    print("=" * 70)

    # Load dataset
    dataset_path = Path("data/processed/chord_dataset.pkl")
    if not dataset_path.exists():
        print("\nERROR: Dataset not found")
        print("Run: python src/data/make_dataset.py")
        return

    print("\nStep 1: Loading dataset...")
    print("-" * 70)
    data = load_dataset(dataset_path)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    label_mapping = data["label_mapping"]

    # Train Logistic Regression
    print("\nStep 2: Training Logistic Regression...")
    print("-" * 70)
    model = train_logistic_regression(X_train, y_train)

    # Evaluate on validation set
    print("\nStep 3: Evaluating on validation set...")
    print("-" * 70)
    val_accuracy = evaluate_model(model, X_val, y_val, label_mapping, "Validation")

    # Evaluate on test set
    print("\nStep 4: Evaluating on test set...")
    print("-" * 70)
    test_accuracy = evaluate_model(model, X_test, y_test, label_mapping, "Test")

    # Final summary
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 70)

    print("\nFinal Results:")
    print(f"  Logistic Regression (Validation): {val_accuracy * 100:.2f}%")
    print(f"  Logistic Regression (Test):       {test_accuracy * 100:.2f}%")

    print("\nModel Characteristics:")
    print("  Model type: Logistic Regression (L2 regularization)")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Number of features: {X_train.shape[1]}")
    print(f"  Number of classes: {len(label_mapping)}")
    print("  Solver: lbfgs (quasi-Newton)")
    print("  Multi-class: multinomial (softmax)")

    # Save results summary
    results_summary = {
        "model": "Logistic Regression",
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "training_samples": len(X_train),
        "features": X_train.shape[1],
        "classes": len(label_mapping),
    }

    output_path = Path("logistic_regression_baseline_results.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(results_summary, f)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
