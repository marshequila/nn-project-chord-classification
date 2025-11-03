"""
Feature engineering for chord classification.
"""

import numpy as np
from typing import List, Dict, Tuple


def chord_to_features(chord: Dict, method: str = "one_hot") -> np.ndarray:
    """
    Convert a chord dictionary to feature vector.

    Args:
        chord: Chord dictionary from parse_midi
        method: 'pitch_class' or 'one_hot'

    Returns:
        Feature vector (numpy array)

    Raises:
        ValueError: If method is not recognized
    """
    inst_notes = []
    for inst_key in ["violin1", "violin2", "viola", "cello"]:
        notes = chord.get(inst_key, [])
        inst_notes.append(notes[0] if notes else -1)

    if method == "pitch_class":
        # Simple: pitch class (0-11) for each instrument
        features = [note % 12 if note != -1 else -1 for note in inst_notes]
        return np.array(features, dtype=np.float32)

    elif method == "one_hot":
        # One-hot encode pitch class for each instrument
        feature_vector = []
        for note in inst_notes:
            pitch_class_vector = [0] * 12
            if note != -1:
                pitch_class_vector[note % 12] = 1
            feature_vector.extend(pitch_class_vector)
        return np.array(feature_vector, dtype=np.float32)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'pitch_class' or 'one_hot'")


def create_label_mapping(unique_labels: List[str]) -> Dict[str, int]:
    """
    Create mapping from chord labels to integers.

    Args:
        unique_labels: List of unique chord labels

    Returns:
        Dictionary mapping label -> integer (sorted alphabetically)
    """
    return {label: idx for idx, label in enumerate(sorted(unique_labels))}


def prepare_dataset(
    chords: List[Dict], label_mapping: Dict[str, int], feature_method: str = "one_hot"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and labels for training.

    Args:
        chords: List of chord dictionaries
        label_mapping: Label to integer mapping
        feature_method: Feature extraction method ('pitch_class' or 'one_hot')

    Returns:
        (X, y) where X is features (n_samples, n_features),
               y is labels (n_samples,)
    """
    X = []
    y = []

    for chord in chords:
        label = chord["label"]
        # Only process chords with known labels
        if label in label_mapping:
            features = chord_to_features(chord, method=feature_method)
            X.append(features)
            y.append(label_mapping[label])

    return np.array(X), np.array(y)
