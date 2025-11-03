"""
Create train/val/test datasets from MIDI files.
"""

import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from sklearn.model_selection import train_test_split
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.parse_midi import extract_chords_from_midi, get_dataset_statistics
from features.build_features import create_label_mapping, prepare_dataset


def process_all_midi_files(midi_dir: str) -> List[Dict]:
    """Process all MIDI files in a directory and extract chords."""
    midi_path = Path(midi_dir)
    midi_files = list(midi_path.glob("*.mid")) + list(midi_path.glob("*.midi"))

    if not midi_files:
        raise ValueError(f"No MIDI files found in {midi_dir}")

    print(f"Found {len(midi_files)} MIDI files\n")

    all_chords = []
    successful_files = 0

    for i, midi_file in enumerate(midi_files, 1):
        try:
            print(
                f"  [{i}/{len(midi_files)}] Processing {midi_file.name[:40]:40}... ",
                end="",
            )
            chords = extract_chords_from_midi(str(midi_file), simplify=True)
            all_chords.extend(chords)
            successful_files += 1
            print(f"OK ({len(chords)} chords)")
        except Exception as e:
            print(f"ERROR: {str(e)[:200]}")
            continue

    failed_files = len(midi_files) - successful_files
    print(f"\nProcessed {successful_files}/{len(midi_files)} files successfully")
    if failed_files > 0:
        print(f"WARNING: {failed_files} files failed (skipped)")

    return all_chords


def filter_rare_chords(all_chords: List[Dict], min_samples: int = 10) -> List[Dict]:
    """Filter out chord types that appear less than min_samples times."""
    # count the label occurrences
    label_counts = {}
    for chord in all_chords:
        label = chord["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    filtered_chords = [
        chord for chord in all_chords if label_counts[chord["label"]] >= min_samples
    ]

    removed = len(all_chords) - len(filtered_chords)
    print(f"  Kept {len(filtered_chords)} / {len(all_chords)} chords")
    if removed > 0:
        print(f"  Removed {removed} chords from rare classes")

    return filtered_chords


def print_statistics(stats: Dict) -> None:
    """Print dataset statistics."""
    print(f"  Total chords: {stats['total_chords']}")
    print(f"  Unique chord types: {stats['unique_labels']}")
    print("\n  Chord distribution:")

    for label, count in sorted(stats["label_distribution"].items()):
        pct = (count / stats["total_chords"]) * 100
        bar = "#" * int(pct / 2)
        print(f"    {label:10} {count:5} ({pct:5.1f}%) {bar}")


def save_dataset(dataset: Dict, output_file: Path, y: np.ndarray) -> None:
    """Save dataset to pickle file."""
    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)
    print(f"  File: {output_file}")


def save_label_mapping(label_mapping: Dict, label_file: Path, y: np.ndarray) -> None:
    """Save label mapping to text file."""
    with open(label_file, "w") as f:
        f.write("Label Mapping\n")
        f.write("=" * 50 + "\n\n")
        for label, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
            count = int(sum(y == idx))
            f.write(f"{idx}: {label:10} ({count} samples)\n")
    print(f"  File: {label_file}")


def main():
    """Main dataset creation pipeline."""

    print("=" * 70)
    print("QUARTET CHORD DATASET CREATION")
    print("=" * 70)

    # Configuration
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"
    min_samples = 10  # Minimum samples per chord type

    # Create processed dir if needed
    Path(processed_data_dir).mkdir(parents=True, exist_ok=True)

    #Extract chords from all MIDI files
    print("\nStep 1: Processing MIDI files...")
    print("-" * 70)
    all_chords = process_all_midi_files(raw_data_dir)
    print(f"\nExtracted {len(all_chords)} total chords")

    if len(all_chords) == 0:
        print("\nERROR: No chords extracted!")
        print("Make sure you have valid MIDI files in data/raw/")
        return

    # Show statistics
    print("\nStep 2: Dataset statistics...")
    print("-" * 70)
    stats = get_dataset_statistics(all_chords)
    print_statistics(stats)

    #Filter rare chords
    print(f"\nStep 3: Filtering rare chords (min {min_samples} samples)...")
    print("-" * 70)
    filtered_chords = filter_rare_chords(all_chords, min_samples)

    # Check if we still have enough data
    if len(filtered_chords) < 100:
        print(f"\nWARNING: Only {len(filtered_chords)} chords after filtering!")
        print("This is too small for training. Try:")
        print("  1. Adding more MIDI files")
        print("  2. Lowering min_samples threshold")
        return

    #Create label mapping
    print("\nStep 4: Creating label mapping...")
    print("-" * 70)
    unique_labels = sorted(set(c["label"] for c in filtered_chords))
    label_mapping = create_label_mapping(unique_labels)

    print(f"  Created mapping with {len(label_mapping)} classes:")
    for label, idx in sorted(label_mapping.items()):
        count = stats["label_distribution"][label]
        print(f"    {idx}: {label:10} ({count} samples)")

    #Prepare features and labels
    print("\nStep 5: Extracting features...")
    print("-" * 70)
    X, y = prepare_dataset(filtered_chords, label_mapping, feature_method="one_hot")
    print(f"  Feature matrix: {X.shape}")
    print(f"  Label vector: {y.shape}")
    print("  Feature type: one-hot encoding (48 features)")

    #Split into train/val/test
    print("\nStep 6: Splitting dataset...")
    print("-" * 70)

    # Check if we have enough samples per class for stratification
    min_class_size = min(int(sum(y == i)) for i in range(len(label_mapping)))
    stratify = None if min_class_size < 2 else y

    if stratify is None:
        print(
            "  WARNING: Some classes have <2 samples. Using random split (no stratification)."
        )

    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Second split: 75% train, 25% val (of the temp set)
    stratify_temp = y_temp if stratify is not None else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=stratify_temp
    )

    print(
        f"  Train: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)"
    )
    print(f"  Val:   {X_val.shape[0]} samples ({X_val.shape[0] / len(X) * 100:.1f}%)")
    print(f"  Test:  {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)")

    # Save datasets
    print("\nStep 7: Saving datasets...")
    print("-" * 70)

    dataset = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "label_mapping": label_mapping,
        "feature_method": "one_hot",
        "num_classes": len(label_mapping),
        "input_size": int(X.shape[1]),
    }

    output_file = Path(processed_data_dir) / "chord_dataset.pkl"
    save_dataset(dataset, output_file, y)

    label_file = Path(processed_data_dir) / "label_mapping.txt"
    save_label_mapping(label_mapping, label_file, y)

    # Final summary
    print("\n" + "=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)
    print("\nDataset summary:")
    print(f"  Input features: {X_train.shape[1]}")
    print(f"  Output classes: {len(label_mapping)}")
    print(f"  Total samples: {len(X)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print("\nFiles created:")
    print(f"  {output_file}")
    print(f"  {label_file}")


if __name__ == "__main__":
    main()
