"""
Plotting and figure generation for chord classification project.

Contains all visualization functions for training results, uncertainty analysis,
and publication-quality figures.

Usage:
    from figures import plot_training_history, plot_uncertainty_analysis
    
    # During training
    plot_training_history(history, 'training_history.png')
    plot_uncertainty_analysis(results, label_mapping, 'uncertainty_analysis.png')
    
    # For publication figures
    python figures.py --model chord_classifier_model.pth --data data/processed/chord_dataset.pkl
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import argparse
from sklearn.metrics import classification_report, confusion_matrix

# Try to import model - handle both relative and absolute imports
try:
    from chord_classifier import ChordClassifier
except ImportError:
    try:
        from src.models.chord_classifier import ChordClassifier
    except ImportError:
        print("Warning: Could not import ChordClassifier")


def set_style():
    """Set consistent plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def plot_uncertainty_analysis(results, label_mapping, save_path='uncertainty_analysis.png'):
    """
    Plot comprehensive uncertainty analysis with 6 subplots.
    
    Args:
        results: Dictionary from evaluate_model_with_uncertainty containing:
                 - predictions, labels, probabilities, uncertainties
                 - predictive_entropy, aleatoric_uncertainty, epistemic_uncertainty
                 - variation_ratio, correct_mask
        label_mapping: Dictionary mapping class names to indices
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    predictions = results['predictions']
    labels = results['labels']
    probabilities = results['probabilities']
    uncertainties = results['uncertainties']
    correct_mask = results['correct_mask']
    
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}
    target_names = [reverse_mapping[i] for i in range(len(label_mapping))]
    
    # 1. Confidence distribution (correct vs incorrect)
    ax = axes[0, 0]
    ax.hist(probabilities[correct_mask], bins=30, alpha=0.6, label='Correct', 
            color='green', density=True)
    ax.hist(probabilities[~correct_mask], bins=30, alpha=0.6, label='Incorrect', 
            color='red', density=True)
    ax.set_xlabel('Confidence (Probability)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Confidence Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Uncertainty distribution (correct vs incorrect)
    ax = axes[0, 1]
    ax.hist(uncertainties[correct_mask], bins=30, alpha=0.6, label='Correct', 
            color='green', density=True)
    ax.hist(uncertainties[~correct_mask], bins=30, alpha=0.6, label='Incorrect', 
            color='red', density=True)
    ax.set_xlabel('Uncertainty (Normalized Entropy)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Uncertainty Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Confidence vs Uncertainty scatter
    ax = axes[0, 2]
    ax.scatter(probabilities[correct_mask], uncertainties[correct_mask], 
              alpha=0.3, s=10, label='Correct', color='green')
    ax.scatter(probabilities[~correct_mask], uncertainties[~correct_mask], 
              alpha=0.5, s=20, label='Incorrect', color='red', marker='x')
    ax.set_xlabel('Confidence', fontweight='bold')
    ax.set_ylabel('Uncertainty', fontweight='bold')
    ax.set_title('Confidence vs Uncertainty', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Per-class confidence
    ax = axes[1, 0]
    class_confidences = [probabilities[labels == i] for i in range(len(target_names))]
    bp = ax.boxplot(class_confidences, tick_labels=target_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Confidence', fontweight='bold')
    ax.set_title('Confidence by Class', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Per-class uncertainty
    ax = axes[1, 1]
    class_uncertainties = [uncertainties[labels == i] for i in range(len(target_names))]
    bp = ax.boxplot(class_uncertainties, tick_labels=target_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
    ax.set_ylabel('Uncertainty', fontweight='bold')
    ax.set_title('Uncertainty by Class', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Uncertainty decomposition comparison
    ax = axes[1, 2]
    metrics = ['Total\n(Pred. Ent.)', 'Aleatoric\n(Data)', 'Epistemic\n(Model)', 'Variation\nRatio']
    correct_means = [
        np.mean(results['predictive_entropy'][correct_mask]),
        np.mean(results['aleatoric_uncertainty'][correct_mask]),
        np.mean(results['epistemic_uncertainty'][correct_mask]),
        np.mean(results['variation_ratio'][correct_mask])
    ]
    incorrect_means = [
        np.mean(results['predictive_entropy'][~correct_mask]),
        np.mean(results['aleatoric_uncertainty'][~correct_mask]),
        np.mean(results['epistemic_uncertainty'][~correct_mask]),
        np.mean(results['variation_ratio'][~correct_mask])
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, correct_means, width, label='Correct', color='green', alpha=0.7)
    ax.bar(x + width/2, incorrect_means, width, label='Incorrect', color='red', alpha=0.7)
    ax.set_ylabel('Mean Value', fontweight='bold')
    ax.set_title('Uncertainty Decomposition', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def plot_confusion_matrix(cm, label_mapping, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix (numpy array)
        label_mapping: Dictionary mapping class names to indices
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}
    class_names = [reverse_mapping[i] for i in range(len(label_mapping))]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def load_model_and_data(model_path, data_path):
    """Load trained model and dataset."""
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    
    print(f"Loading data: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get results
    test_accuracy = checkpoint.get('test_accuracy', 0)
    history = checkpoint.get('history', {})
    
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Test samples: {len(data['y_test'])}")
    
    return checkpoint, data, test_accuracy, history


def generate_per_class_figure(data, checkpoint, save_path='per_class_performance.png'):
    """
    Generate per-class performance analysis figure.
    Shows precision, recall, F1-score, and class distribution.
    
    Args:
        data: Dataset dictionary
        checkpoint: Model checkpoint
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    print("Generating per-class performance figure...")
    
    # Load model and get predictions
    model = ChordClassifier(
        input_size=checkpoint['input_size'],
        hidden_sizes=checkpoint.get('architecture', [128, 64, 32, 16]),
        num_classes=checkpoint['num_classes'],
        dropout_rate=checkpoint.get('dropout_rate', 0.1)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get predictions
    X_test = torch.FloatTensor(data['X_test'])
    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = torch.max(outputs, 1)
    
    predictions = predictions.numpy()
    y_test = data['y_test']
    label_mapping = data['label_mapping']
    
    # Calculate metrics
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}
    class_names = [reverse_mapping[i] for i in range(len(label_mapping))]
    
    # Get classification report
    report = classification_report(y_test, predictions, 
                                   target_names=class_names, 
                                   output_dict=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Classification metrics by class
    ax = axes[0, 0]
    x = np.arange(len(class_names))
    width = 0.25
    
    precisions = [report[name]['precision'] for name in class_names]
    recalls = [report[name]['recall'] for name in class_names]
    f1_scores = [report[name]['f1-score'] for name in class_names]
    
    ax.bar(x - width, precisions, width, label='Precision', color='#3498db', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', color='#2ecc71', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c', alpha=0.8)
    
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics by Class', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Class distribution
    ax = axes[0, 1]
    class_counts = [np.sum(y_test == i) for i in range(len(class_names))]
    total_samples = len(y_test)
    colors = ['#3498db', '#e74c3c', '#f39c12']
    
    bars = ax.bar(class_names, class_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        percentage = 100 * count / total_samples
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Test Set Class Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Recall breakdown
    ax = axes[1, 0]
    correct_counts = []
    missed_counts = []
    
    for i in range(len(class_names)):
        class_mask = y_test == i
        correct = np.sum((predictions == y_test) & class_mask)
        missed = np.sum((predictions != y_test) & class_mask)
        correct_counts.append(correct)
        missed_counts.append(missed)
    
    x = np.arange(len(class_names))
    width = 0.6
    
    ax.bar(x, correct_counts, width, label='Correctly Identified',
           color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=2)
    ax.bar(x, missed_counts, width, bottom=correct_counts, label='Missed',
           color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Recall Breakdown', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Confusion matrix
    ax = axes[1, 1]
    cm = confusion_matrix(y_test, predictions)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted',
           ylabel='True')
    ax.set_title('Confusion Matrix', fontweight='bold')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def generate_model_comparison_figure(checkpoint, 
                                     lr_accuracy=50.3,
                                     lr_recall=None,
                                     save_path='model_comparison.png'):
    """
    Generate model comparison figure (Neural Network vs Baseline).
    
    Args:
        checkpoint: Model checkpoint
        lr_accuracy: Logistic regression baseline accuracy
        lr_recall: Dictionary of per-class recall for LR (optional)
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    print("Generating model comparison figure...")
    
    if lr_recall is None:
        lr_recall = {'Major': 0.77, 'Other': 0.0, 'minor': 0.36}
    
    # Neural network results
    nn_accuracy = checkpoint.get('test_accuracy', 83.0)
    nn_recall = {'Major': 0.88, 'Other': 0.63, 'minor': 0.83}
    
    # Model parameters
    architecture = checkpoint.get('architecture', [128, 64, 32, 16])
    input_size = checkpoint.get('input_size', 48)
    num_classes = checkpoint.get('num_classes', 3)
    
    # Calculate NN parameters
    nn_params = input_size * architecture[0] + architecture[0]
    for i in range(len(architecture) - 1):
        nn_params += architecture[i] * architecture[i+1] + architecture[i+1]
    nn_params += architecture[-1] * num_classes + num_classes
    
    lr_params = input_size * num_classes + num_classes
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Test accuracy comparison
    ax = axes[0, 0]
    models = ['Random', 'Majority\nClass', 'Logistic\nRegression', 'Neural\nNetwork']
    accuracies = [39.0, 47.6, lr_accuracy, nn_accuracy]
    colors = ['#95a5a6', '#7f8c8d', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Model Comparison: Test Accuracy', fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Per-class recall comparison
    ax = axes[0, 1]
    classes = list(nn_recall.keys())
    lr_recalls = [lr_recall.get(c, 0) for c in classes]
    nn_recalls = [nn_recall.get(c, 0) for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    ax.bar(x - width/2, lr_recalls, width, label='Logistic Regression',
           color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=2)
    ax.bar(x + width/2, nn_recalls, width, label='Neural Network',
           color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Recall')
    ax.set_title('Per-Class Recall Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Model complexity
    ax = axes[1, 0]
    models_short = ['LR', 'NN']
    params = [lr_params, nn_params]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax.bar(models_short, params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Model Complexity', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Accuracy vs complexity scatter
    ax = axes[1, 1]
    
    ax.scatter([lr_params], [lr_accuracy], s=300, 
              color='#e74c3c', alpha=0.6, edgecolor='black', linewidth=2,
              label='Logistic Regression')
    ax.scatter([nn_params], [nn_accuracy], s=300,
              color='#2ecc71', alpha=0.6, edgecolor='black', linewidth=2,
              label='Neural Network')
    
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Accuracy vs Model Complexity', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def main():
    """Main function to generate publication figures."""
    parser = argparse.ArgumentParser(description='Generate figures for chord classification')
    parser.add_argument('--model', type=str, default='models/chord_classifier_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='data/processed/chord_dataset.pkl',
                       help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for figures')
    parser.add_argument('--lr-accuracy', type=float, default=50.3,
                       help='Logistic regression baseline accuracy')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set plotting style
    set_style()
    
    print("\n" + "="*70)
    print("Figure Generation for Chord Classification")
    print("="*70)
    
    # Load model and data
    checkpoint, data, test_accuracy, history = load_model_and_data(
        args.model, args.data
    )
    
    print("\nGenerating figures...")
    
    # Generate figures
    generate_per_class_figure(
        data, checkpoint, 
        save_path=f'{args.output_dir}/per_class_performance.png'
    )
    
    generate_model_comparison_figure(
        checkpoint,
        lr_accuracy=args.lr_accuracy,
        save_path=f'{args.output_dir}/model_comparison.png'
    )
    
    # If history available, plot training curves
    if history:
        plot_training_history(history, f'{args.output_dir}/training_history.png')
    
    print("\n" + "="*70)
    print("All figures generated")
    print(f"Saved to: {args.output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()