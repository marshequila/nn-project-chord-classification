"""
MIDI parsing functions for extracting chords from quartet files.
"""

import pretty_midi
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from music21 import chord as music21_chord
from music21 import pitch as music21_pitch


def load_midi(filepath: str) -> pretty_midi.PrettyMIDI:
    """
    Load a MIDI file.

    Args:
        filepath: Path to MIDI file

    Returns:
        PrettyMIDI object

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"MIDI file not found: {filepath}")

    return pretty_midi.PrettyMIDI(filepath)


def get_notes_at_time(
    instrument: pretty_midi.Instrument, time_point: float, tolerance: float = 0.01
) -> List[int]:
    """
    Get all notes active at a specific time point.

    Args:
        instrument: PrettyMIDI instrument object
        time_point: Time in seconds
        tolerance: Time tolerance in seconds

    Returns:
        List of MIDI pitch numbers
    """
    active_notes = []
    for note in instrument.notes:
        # Check if note is playing at this time
        if note.start <= time_point <= note.end:
            active_notes.append(note.pitch)

    return active_notes


def classify_chord(notes: List[int]) -> str:
    """
    Classify chord using music21 library.
    Much more accurate than manual templates!
    """
    if len(notes) < 2:
        return "Single Note"

    if len(notes) == 2:
        return "Interval"

    try:
        # Convert MIDI numbers to music21 pitches
        pitches = [music21_pitch.Pitch(midi=n) for n in notes]

        # Create chord object
        c = music21_chord.Chord(pitches)

        # Get chord name
        root = c.root().name
        quality = c.commonName


        return f"{root} {quality}"

    except Exception as e:
        return "Unknown"


def simplify_chord_label(label: str) -> str:
    """
    Simplify music21 chord labels to Major/minor/Other.

    Args:
        label: Chord label from music21 (e.g., "C major triad", "G dominant seventh")

    Returns:
        Simplified label: "Major", "minor", or "Other"
    """
    label_lower = label.lower()

    # Major variants
    if "major" in label_lower and "minor" not in label_lower:
        return "Major"

    # Minor variants
    if "minor" in label_lower:
        return "minor"

    # Diminished (sounds minor-ish)
    if "diminished" in label_lower:
        return "minor"

    # Augmented (sounds major-ish)
    if "augmented" in label_lower:
        return "Major"

    # Dominant (major with flat 7)
    if "dominant" in label_lower:
        return "Major"

    # Suspended (ambiguous, call it major)
    if "suspended" in label_lower:
        return "Major"

    # Everything else
    return "Other"


def extract_chords_from_midi(
    midi_path: str,
    beat_duration: float = 0.5,
    min_notes: int = 3,
    simplify: bool = True,
) -> List[Dict]:  
    """
    Extract all chords from a quartet MIDI file.

    Args:
        midi_path: Path to MIDI file
        beat_duration: Time interval for sampling (seconds)
        min_notes: Minimum notes required for valid chord
        simplify: If True, simplify labels to Major/minor/Other

    Returns:
        List of chord dictionaries
    """
    midi = load_midi(midi_path)

    if len(midi.instruments) < 4:
        raise ValueError(f"Expected 4 instruments, found {len(midi.instruments)}")

    instruments = midi.instruments[:4]
    duration = midi.get_end_time()

    # Sample at regular intervals
    num_samples = int(duration / beat_duration)
    chords = []

    for i in range(num_samples):
        # at what second does the chord start
        time_point = i * beat_duration

        # Get notes from each instrument
        violin1_notes = get_notes_at_time(instruments[0], time_point)
        violin2_notes = get_notes_at_time(instruments[1], time_point)
        viola_notes = get_notes_at_time(instruments[2], time_point)
        cello_notes = get_notes_at_time(instruments[3], time_point)

        all_notes = violin1_notes + violin2_notes + viola_notes + cello_notes

        # Only keep if we have enough notes
        if len(all_notes) >= min_notes:
            full_label = classify_chord(all_notes)

            if simplify:
                chord_label = simplify_chord_label(full_label)
            else:
                chord_label = full_label

            chords.append(
                {
                    "time": time_point,
                    "notes": all_notes,
                    "label": chord_label,
                    "full_label": full_label,
                    "violin1": violin1_notes,
                    "violin2": violin2_notes,
                    "viola": viola_notes,
                    "cello": cello_notes,
                    "source_file": Path(midi_path).name,
                }
            )

    return chords


def get_dataset_statistics(chords: List[Dict]) -> Dict:
    """
    Calculate statistics about extracted chords.

    Args:
        chords: List of chord dictionaries

    Returns:
        Dictionary with statistics
    """
    from collections import Counter

    labels = [c["label"] for c in chords]
    label_counts = Counter(labels)

    return {
        "total_chords": len(chords),
        "unique_labels": len(label_counts),
        "label_distribution": dict(label_counts),
        "most_common": label_counts.most_common(5),
    }



if __name__ == "__main__":
    import sys

    test_file = "data/raw/quartet_1_1_(c)edwards.mid"

    if not Path(test_file).exists():
        print(f"File not found: {test_file}")
        print("Create a test MIDI file first or specify a different path")
        sys.exit(1)

    print(f"Processing {test_file}...")
    print("=" * 70)

    try:
        # Extract chords with simplified labels
        print("\nEXTRACTING WITH SIMPLIFIED LABELS (Major/minor/Other)")
        print("-" * 70)
        chords = extract_chords_from_midi(test_file, simplify=True)

        # Show statistics
        stats = get_dataset_statistics(chords)

        print(f"\nSuccessfully extracted {stats['total_chords']} chords")
        print(f"Found {stats['unique_labels']} unique chord types")

        print("\nChord distribution:")
        for label, count in sorted(stats["label_distribution"].items()):
            percentage = (count / stats["total_chords"]) * 100
            bar = "#" * int(percentage / 2)
            print(f"   {label:10} {count:4} ({percentage:5.1f}%) {bar}")

        # Show first 10 chords with both labels
        print("\nFirst 10 chords (showing simplified + original):")
        print("-" * 70)
        for i, chord in enumerate(chords[:10]):
            notes_str = ", ".join(
                [
                    pretty_midi.note_number_to_name(n)
                    for n in sorted(set(chord["notes"]))[:4]
                ]
            )  # First 4 unique notes
            simplified = chord["label"]
            original = chord.get("full_label", "N/A")
            print(
                f"   {i + 1:2}. Time {chord['time']:5.1f}s: {simplified:6} (was: {original:20}) [{notes_str}]"
            )

        # Check class balance
        print("\nCLASS BALANCE CHECK:")
        print("-" * 70)
        total = stats["total_chords"]
        for label in ["Major", "minor", "Other"]:
            count = stats["label_distribution"].get(label, 0)
            pct = (count / total) * 100 if total > 0 else 0

            if 25 <= pct <= 40:
                status = "Good balance"
            elif 15 <= pct < 25 or 40 < pct <= 50:
                status = "Slight imbalance"
            else:
                status = "Needs rebalancing"

            print(f"   {label:10}: {count:4} ({pct:5.1f}%) {status}")

    except Exception as exc:
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
