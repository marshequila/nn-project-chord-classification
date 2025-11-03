"""
Check all MIDI files are valid and have 4 instruments.
"""
import pretty_midi
from pathlib import Path

midi_dir = Path("data/raw")
midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))

print(f"Found {len(midi_files)} MIDI files in data/raw/")
print("=" * 70)

valid_files = []
total_chords_estimate = 0

for i, midi_file in enumerate(midi_files, 1):
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_file))
        num_instruments = len(midi.instruments)
        duration = midi.get_end_time()
        
        # Rough estimate: ~1 chord per second
        estimated_chords = int(duration / 0.5)  # 0.5s per chord
        total_chords_estimate += estimated_chords
        
        if num_instruments >= 4:
            status = "OK"
            valid_files.append(midi_file)
        else:
            status = "WARNING"
        
        print(f"{status} [{i}] {midi_file.name[:45]:45}")
        print(f"     Instruments: {num_instruments}, Duration: {duration:.1f}s, Est. chords: ~{estimated_chords}")
        
    except Exception as e:
        print(f"ERROR [{i}] {midi_file.name[:45]:45}")
        print(f"     Error: {str(e)[:50]}")

print("\n" + "=" * 70)
print(f"Valid files: {len(valid_files)}/{len(midi_files)}")
print(f"Estimated total chords: ~{total_chords_estimate}")

