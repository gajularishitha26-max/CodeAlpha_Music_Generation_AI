"""
scripts/create_demo_midi.py
===========================

Generate a few TINY, SYNTHETIC demo MIDI files so the training / generation
pipeline can be smoke-tested end-to-end without downloading a real dataset.

IMPORTANT: These files are artificially constructed short melodies / scales.
They are NOT real music and exist ONLY to prove the pipeline runs. For any
meaningful results, replace them with a real MIDI dataset in data/midi/.

Part of the CodeAlpha AI Internship - Task 3 (Music Generation with AI).
"""

import os

# data/midi directory relative to the project root (parent of this script's dir).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "midi")


def _build_melodies():
    """Return a list of ``(filename, [pitch_names])`` demo melodies.

    Each melody is a short, deterministic sequence of pitch names. Filenames
    are prefixed with ``demo_`` to make their synthetic nature obvious.
    """
    return [
        # A simple ascending + descending C-major scale.
        (
            "demo_scale_1.mid",
            [
                "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5",
                "B4", "A4", "G4", "F4", "E4", "D4", "C4",
            ],
        ),
        # A short arpeggio-style melody.
        (
            "demo_melody_2.mid",
            [
                "C4", "E4", "G4", "C5", "G4", "E4",
                "F4", "A4", "C5", "A4", "F4", "C4",
            ],
        ),
        # A little stepwise tune with some repetition.
        (
            "demo_melody_3.mid",
            [
                "E4", "E4", "F4", "G4", "G4", "F4", "E4", "D4",
                "C4", "C4", "D4", "E4", "E4", "D4", "D4",
            ],
        ),
    ]


def create_demo_midi(data_dir=DATA_DIR):
    """Write the synthetic demo MIDI files into ``data_dir``.

    Parameters
    ----------
    data_dir : str
        Destination directory for the demo files (created if missing).

    Returns
    -------
    list[str]
        Paths of the demo MIDI files that were written.
    """
    # music21 imported here so the module stays importable without it.
    from music21 import note, stream

    os.makedirs(data_dir, exist_ok=True)

    written = []
    for filename, pitches in _build_melodies():
        melody = stream.Stream()
        offset = 0.0
        for pitch_name in pitches:
            new_note = note.Note(pitch_name)
            new_note.quarterLength = 0.5
            new_note.offset = offset
            melody.append(new_note)
            offset += 0.5

        out_path = os.path.join(data_dir, filename)
        melody.write("midi", fp=out_path)
        written.append(out_path)

    return written


def main():
    """CLI entry point: create the demo files and report where they went."""
    print("Creating synthetic demo MIDI files (for smoke-testing only)...")
    try:
        written = create_demo_midi()
    except ImportError:
        print(
            "Error: 'music21' is not installed. Install dependencies with: "
            "pip install -r requirements.txt"
        )
        return

    for path in written:
        print("  wrote: {}".format(path))
    print("Done. {} demo file(s) written to: {}".format(len(written), DATA_DIR))


if __name__ == "__main__":
    main()
