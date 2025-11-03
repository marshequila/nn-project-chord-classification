"""
Training script for chord classification with MC Dropout uncertainty estimation.
  - Train/Validation/Test split (60/20/20)
  - Fixed hyperparameters: dropout=0.1, lr=0.001
  - Final evaluation on TEST set (once at the end)
  - MC Dropout for uncertainty estimation
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from chord_classifier import ChordClassifier, count_parameters
from figures import plot_training_history, plot_uncertainty_analysis


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


def create_dataloaders(data, batch_size=32):
    """Create PyTorch DataLoaders for training, validation, and test sets."""

    X_train = torch.FloatTensor(data["X_train"])
    y_train = torch.LongTensor(data["y_train"])

    X_val = torch.FloatTensor(data["X_val"])
    y_val = torch.LongTensor(data["y_val"])

    X_test = torch.FloatTensor(data["X_test"])
    y_test = torch.LongTensor(data["y_test"])

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    patience=25,
    verbose=True,
):
    """
    Train the model with early stopping.

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        device: Device to train on
        patience: Early stopping patience
        verbose: Whether to print progress

    Returns:
        Dictionary with training history
    """

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    if verbose:
        print()

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print progress
        if verbose:
            print(
                f"Epoch {epoch + 1:3d}/{num_epochs}: "
                f"loss {train_loss:.4f}/{val_loss:.4f} | "
                f"acc {train_acc:.2f}%/{val_acc:.2f}%"
            )

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            if verbose:
                print(
                    f"\nEarly stopping at epoch {epoch + 1} (best val loss: {best_val_loss:.4f})"
                )
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history


def enable_dropout(model):
    """
    Enable dropout layers during inference for MC Dropout.

    This function sets all dropout layers to training mode while keeping
    batch normalization (if any) in eval mode.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(model, X, device, n_samples=100):
    """
    Make predictions using MC Dropout for uncertainty estimation.

    Uncertainty Decomposition:
        - Total Uncertainty = Aleatoric + Epistemic
        - Predictive Entropy (total) = Expected Entropy (aleatoric) + Mutual Information (epistemic)

    Args:
        model: Trained neural network model
        X: Input tensor (batch_size, input_size)
        device: Device to run on
        n_samples: Number of stochastic forward passes

    Returns:
        predictions: Most likely class for each input (batch_size,)
        probabilities: Mean probability for predicted class (batch_size,)
        uncertainties: Predictive uncertainty (batch_size,)
        all_probs: Mean probabilities for all classes (batch_size, num_classes)
        prob_std: Standard deviation of probabilities (batch_size, num_classes)
        predictive_entropy: Total predictive entropy (batch_size,)
        aleatoric_uncertainty: Data/irreducible uncertainty (batch_size,)
        epistemic_uncertainty: Model/knowledge uncertainty (batch_size,)
        variation_ratio: Variation ratio (batch_size,)
    """
    model.eval()
    X = X.to(device)

    # Storage for multiple predictions
    batch_size = X.shape[0]
    num_classes = model.num_classes
    predictions_list = []

    # Perform multiple stochastic forward passes
    with torch.no_grad():
        for _ in range(n_samples):
            # Enable dropout for this forward pass
            enable_dropout(model)

            # Forward pass
            outputs = model(X)
            probs = F.softmax(outputs, dim=1)  
            predictions_list.append(probs.cpu().numpy())

    # Stack all predictions: (n_samples, batch_size, num_classes)
    predictions_array = np.stack(predictions_list, axis=0)

    # Calculate statistics
    # Mean probabilities across MC samples
    mean_probs = np.mean(predictions_array, axis=0)  
    # Standard deviation of probabilities
    prob_std = np.std(predictions_array, axis=0)  

    # Predicted class (highest mean probability)
    predictions = np.argmax(mean_probs, axis=1)  

    # Confidence: mean probability of predicted class
    probabilities = mean_probs[np.arange(batch_size), predictions]

    # === UNCERTAINTY DECOMPOSITION ===
    epsilon = 1e-10

    # 1. TOTAL UNCERTAINTY
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=1)

    # 2. ALEATORIC UNCERTAINTY
    aleatoric_uncertainty = np.mean(
        -np.sum(predictions_array * np.log(predictions_array + epsilon), axis=2), axis=0
    )

    # 3. EPISTEMIC UNCERTAINTY
    epistemic_uncertainty = predictive_entropy - aleatoric_uncertainty

    # 4. Variation ratio (fraction of predictions that differ from mode)
    predicted_classes = np.argmax(predictions_array, axis=2)
    mode_counts = np.array(
        [np.max(np.bincount(predicted_classes[:, i])) for i in range(batch_size)]
    )
    variation_ratio = 1 - (mode_counts / n_samples)

    # Combined uncertainty score (normalized predictive entropy)
    uncertainties = predictive_entropy / np.log(num_classes)

    return (
        predictions,
        probabilities,
        uncertainties,
        mean_probs,
        prob_std,
        predictive_entropy,
        aleatoric_uncertainty,
        epistemic_uncertainty,
        variation_ratio,
    )


def evaluate_model(model, test_loader, device, label_mapping):
    """
    Evaluate model on test set and print detailed metrics.
    Standard evaluation without MC Dropout.
    """
    model.eval()
    all_predictions = []
    all_labels = []

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

    # Calculate accuracy
    accuracy = 100 * np.sum(all_predictions == all_labels) / len(all_labels)

    print(f"\nTest set accuracy: {accuracy:.2f}%")

    # Create reverse mapping (index -> label)
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}
    target_names = [reverse_mapping[i] for i in range(len(label_mapping))]

    # Classification report
    print("\nDetailed metrics:")
    print(classification_report(all_labels, all_predictions, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion matrix:")
    print(f"{'':10} {' '.join([f'{name:10}' for name in target_names])}")
    for i, row in enumerate(cm):
        print(f"{target_names[i]:10} {' '.join([f'{val:10}' for val in row])}")

    return accuracy, all_predictions, all_labels, cm


def evaluate_model_with_uncertainty(
    model, test_loader, device, label_mapping, n_samples=100
):
    """
    Evaluate model on test set with MC Dropout uncertainty estimation.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        label_mapping: Dictionary mapping class names to indices
        n_samples: Number of MC Dropout samples for uncertainty estimation

    Returns:
        Dictionary with predictions, uncertainties, and metrics
    """
    print(f"Running MC Dropout with {n_samples} forward passes per sample...")

    model.eval()

    all_predictions = []
    all_probabilities = []
    all_uncertainties = []
    all_labels = []
    all_mean_probs = []
    all_prob_std = []
    all_pred_entropy = []
    all_aleatoric = []
    all_epistemic = []
    all_variation_ratio = []

    # Process in batches
    for batch_idx, (batch_X, batch_y) in enumerate(test_loader):
        # MC Dropout predictions
        (
            predictions,
            probabilities,
            uncertainties,
            mean_probs,
            prob_std,
            pred_entropy,
            aleatoric,
            epistemic,
            var_ratio,
        ) = mc_dropout_predict(model, batch_X, device, n_samples=n_samples)

        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)
        all_uncertainties.extend(uncertainties)
        all_labels.extend(batch_y.numpy())
        all_mean_probs.append(mean_probs)
        all_prob_std.append(prob_std)
        all_pred_entropy.extend(pred_entropy)
        all_aleatoric.extend(aleatoric)
        all_epistemic.extend(epistemic)
        all_variation_ratio.extend(var_ratio)

        if (batch_idx + 1) % 20 == 0:
            print(
                f"  {(batch_idx + 1) * test_loader.batch_size}/{len(test_loader.dataset)} samples"
            )

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_uncertainties = np.array(all_uncertainties)
    all_labels = np.array(all_labels)
    all_mean_probs = np.vstack(all_mean_probs)
    all_prob_std = np.vstack(all_prob_std)
    all_pred_entropy = np.array(all_pred_entropy)
    all_aleatoric = np.array(all_aleatoric)
    all_epistemic = np.array(all_epistemic)
    all_variation_ratio = np.array(all_variation_ratio)

    # Calculate accuracy
    accuracy = 100 * np.sum(all_predictions == all_labels) / len(all_labels)

    print(f"\nAccuracy: {accuracy:.2f}%")

    # Uncertainty statistics
    print("\nUncertainty statistics:")
    print(
        f"  Confidence: {np.mean(all_probabilities):.4f} +/- {np.std(all_probabilities):.4f}"
    )
    print(
        f"  Total uncertainty: {np.mean(all_uncertainties):.4f} +/- {np.std(all_uncertainties):.4f}"
    )
    print("\nDecomposition:")
    print(
        f"  Predictive entropy: {np.mean(all_pred_entropy):.4f} +/- {np.std(all_pred_entropy):.4f}"
    )
    print(
        f"  Aleatoric (data):   {np.mean(all_aleatoric):.4f} +/- {np.std(all_aleatoric):.4f}"
    )
    print(
        f"  Epistemic (model):  {np.mean(all_epistemic):.4f} +/- {np.std(all_epistemic):.4f}"
    )
    print(
        f"  Variation ratio:    {np.mean(all_variation_ratio):.4f} +/- {np.std(all_variation_ratio):.4f}"
    )

    # Verify decomposition
    total_check = np.mean(all_aleatoric) + np.mean(all_epistemic)
    print(
        f"\nCheck: Aleatoric + Epistemic = {total_check:.4f} (expected: {np.mean(all_pred_entropy):.4f})"
    )

    # Analyze correct vs incorrect predictions
    correct_mask = all_predictions == all_labels
    incorrect_mask = ~correct_mask

    if np.sum(correct_mask) > 0:
        print(f"\nCorrect predictions ({np.sum(correct_mask)}):")
        print(f"  Confidence: {np.mean(all_probabilities[correct_mask]):.4f}")
        print(f"  Uncertainty: {np.mean(all_uncertainties[correct_mask]):.4f}")

    if np.sum(incorrect_mask) > 0:
        print(f"\nIncorrect predictions ({np.sum(incorrect_mask)}):")
        print(f"  Confidence: {np.mean(all_probabilities[incorrect_mask]):.4f}")
        print(f"  Uncertainty: {np.mean(all_uncertainties[incorrect_mask]):.4f}")

    # Per-class analysis
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}
    target_names = [reverse_mapping[i] for i in range(len(label_mapping))]

    print("\nPer-class analysis:")
    for class_idx, class_name in enumerate(target_names):
        class_mask = all_labels == class_idx
        if np.sum(class_mask) > 0:
            class_correct = np.sum((all_predictions == all_labels) & class_mask)
            class_total = np.sum(class_mask)
            class_acc = 100 * class_correct / class_total

            print(f"\n{class_name}: {class_acc:.2f}% ({class_correct}/{class_total})")
            print(f"  Confidence: {np.mean(all_probabilities[class_mask]):.4f}")
            print(f"  Total unc.: {np.mean(all_uncertainties[class_mask]):.4f}")
            print(f"  Aleatoric:  {np.mean(all_aleatoric[class_mask]):.4f}")
            print(f"  Epistemic:  {np.mean(all_epistemic[class_mask]):.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion matrix:")
    print(f"{'':10} {' '.join([f'{name:10}' for name in target_names])}")
    for i, row in enumerate(cm):
        print(f"{target_names[i]:10} {' '.join([f'{val:10}' for val in row])}")

    # Classification report
    print("\nDetailed metrics:")
    print(classification_report(all_labels, all_predictions, target_names=target_names))

    # Return results
    results = {
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
        "probabilities": all_probabilities,
        "uncertainties": all_uncertainties,
        "mean_probs": all_mean_probs,
        "prob_std": all_prob_std,
        "predictive_entropy": all_pred_entropy,
        "aleatoric_uncertainty": all_aleatoric,
        "epistemic_uncertainty": all_epistemic,
        "variation_ratio": all_variation_ratio,
        "confusion_matrix": cm,
        "correct_mask": correct_mask,
    }

    return results


def main():
    """Main training pipeline with MC Dropout uncertainty estimation."""

    print("\n" + "=" * 70)
    print("Chord Classification Training")
    print("=" * 70)
    print("\nConfiguration:")
    print("  Split: 60/20/20 (train/val/test)")
    print("  Dropout: 0.1 | Learning rate: 0.001")
    print("  MC Dropout samples: 100")
    print("=" * 70)

    # Configuration
    dataset_path = "data/processed/chord_dataset.pkl"
    batch_size = 32
    num_epochs = 100
    patience = 15
    learning_rate = 0.001
    dropout_rate = 0.1
    architecture = [64, 32, 16]
    mc_samples = 100 

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load dataset
    print("\nLoading dataset...")
    data = load_dataset(dataset_path)

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(data, batch_size)
    print(f"Batch size: {batch_size}")
    print(
        f"Batches: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test"
    )

    # Initialize model
    print("\nInitializing model...")
    print(
        f"Architecture: {data['input_size']} -> {architecture} -> {data['num_classes']}"
    )
    print(f"Dropout: {dropout_rate} | LR: {learning_rate}")

    model = ChordClassifier(
        input_size=data["input_size"],
        hidden_sizes=architecture,
        num_classes=data["num_classes"],
        dropout_rate=dropout_rate,
    )
    model = model.to(device)
    print(f"Total parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print(f"\nTraining (max {num_epochs} epochs, early stop patience={patience})...")

    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs,
        device,
        patience,
        verbose=True,
    )

    # Standard evaluation on TEST set
    print("\n" + "-" * 70)
    print("Evaluating on test set...")
    print("-" * 70)

    test_accuracy, predictions, labels, cm = evaluate_model(
        model, test_loader, device, data["label_mapping"]
    )

    # MC Dropout evaluation for uncertainty
    print("\n" + "-" * 70)
    print(f"Running MC Dropout uncertainty estimation ({mc_samples} samples)...")
    print("-" * 70)

    uncertainty_results = evaluate_model_with_uncertainty(
        model, test_loader, device, data["label_mapping"], n_samples=mc_samples
    )

    # Save results
    print("\nSaving results...")

    # Plot training history
    plot_training_history(history, save_path="training_history.png")

    # Plot uncertainty analysis
    plot_uncertainty_analysis(
        uncertainty_results, data["label_mapping"], save_path="uncertainty_analysis.png"
    )

    # Save final model with complete metadata
    model_path = "chord_classifier_model.pth"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "label_mapping": data["label_mapping"],
        "input_size": data["input_size"],
        "num_classes": data["num_classes"],
        "architecture": architecture,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "history": history,
        "test_accuracy": test_accuracy,
        "test_accuracy_mc": uncertainty_results["accuracy"],
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "mc_dropout_samples": mc_samples,
        "uncertainty_stats": {
            "mean_confidence": float(np.mean(uncertainty_results["probabilities"])),
            "mean_uncertainty": float(np.mean(uncertainty_results["uncertainties"])),
            "mean_pred_entropy": float(
                np.mean(uncertainty_results["predictive_entropy"])
            ),
            "mean_aleatoric": float(
                np.mean(uncertainty_results["aleatoric_uncertainty"])
            ),
            "mean_epistemic": float(
                np.mean(uncertainty_results["epistemic_uncertainty"])
            ),
            "mean_variation_ratio": float(
                np.mean(uncertainty_results["variation_ratio"])
            ),
        },
    }

    torch.save(checkpoint, model_path)
    print(f"Model saved: {model_path}")

    # Save uncertainty results separately
    uncertainty_path = "uncertainty_results.pkl"
    with open(uncertainty_path, "wb") as f:
        pickle.dump(uncertainty_results, f)
    print(f"Uncertainty results saved: {uncertainty_path}")

    # Verify checkpoint
    verify_ckpt = torch.load(model_path, weights_only=False)
    print("\nCheckpoint contents:")
    print(f"  Architecture: {verify_ckpt['architecture']}")
    print(
        f"  Dropout: {verify_ckpt['dropout_rate']} | LR: {verify_ckpt['learning_rate']}"
    )
    print(
        f"  Test accuracy: {verify_ckpt['test_accuracy']:.2f}% (standard), {verify_ckpt['test_accuracy_mc']:.2f}% (MC)"
    )
    print(
        f"  Uncertainty - Total: {verify_ckpt['uncertainty_stats']['mean_uncertainty']:.4f}, "
        f"Aleatoric: {verify_ckpt['uncertainty_stats']['mean_aleatoric']:.4f}, "
        f"Epistemic: {verify_ckpt['uncertainty_stats']['mean_epistemic']:.4f}"
    )

    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print("\nAccuracy:")
    print(f"  Train: {history['train_acc'][-1]:.2f}%")
    print(f"  Val:   {history['val_acc'][-1]:.2f}%")
    print(
        f"  Test:  {test_accuracy:.2f}% (standard) | {uncertainty_results['accuracy']:.2f}% (MC)"
    )
    print("\nUncertainty breakdown:")
    print(f"  Confidence:  {np.mean(uncertainty_results['probabilities']):.4f}")
    print(f"  Total:       {np.mean(uncertainty_results['predictive_entropy']):.4f}")
    print(
        f"  Aleatoric:   {np.mean(uncertainty_results['aleatoric_uncertainty']):.4f} (data)"
    )
    print(
        f"  Epistemic:   {np.mean(uncertainty_results['epistemic_uncertainty']):.4f} (model)"
    )
    print(f"\nTrained for {len(history['train_acc'])} epochs")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
