"""
generate_music.py
=================

Generate a new MIDI file from a trained LSTM model. Part of the CodeAlpha AI
Internship - Task 3 (Music Generation with AI).

The script seeds the network with a starting sequence (random by default, or
sampled from the real dataset with ``--seed-from-data``), then repeatedly
predicts the next token, decodes each token into a music21 ``Note`` or
``Chord``, and writes the result to a MIDI file.

Usage
-----
    python generate_music.py --length 200

Heavy imports (TensorFlow, music21, numpy) live inside ``main()`` so the file
imports and py_compiles cleanly without those packages installed.
"""

import argparse
import os
import sys

import midi_processor


# Time (in quarter-length units) placed between successive generated notes.
OFFSET_STEP = 0.5


def parse_args(argv=None):
    """Parse command line arguments for the generation script."""
    parser = argparse.ArgumentParser(
        description="Generate music with a trained LSTM model."
    )
    parser.add_argument(
        "--model",
        default=os.path.join("models", "music_lstm.keras"),
        help="Path to the trained Keras model.",
    )
    parser.add_argument(
        "--vocab",
        default=os.path.join("models", "vocab.json"),
        help="Path to the vocabulary JSON file.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=200,
        help="Number of notes/chords to generate (default: 200).",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("outputs", "generated_music.mid"),
        help="Output MIDI file path.",
    )
    parser.add_argument(
        "--seed-from-data",
        action="store_true",
        help="Seed the sequence from real MIDI data instead of random tokens.",
    )
    parser.add_argument(
        "--data-dir",
        default=midi_processor.DATA_DIR,
        help="Dataset directory (used only with --seed-from-data).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=midi_processor.SEQUENCE_LENGTH,
        help="Input sequence length the model was trained with (default: 32).",
    )
    return parser.parse_args(argv)


def _build_seed(int_to_note, sequence_length, seed_from_data, data_dir):
    """Create the initial integer pattern used to prime generation.

    Returns a list of integer token ids of length ``sequence_length``. When
    ``seed_from_data`` is set and the dataset yields enough tokens, a real
    slice of the corpus is used; otherwise a random pattern is generated.
    """
    import random

    n_vocab = len(int_to_note)
    note_to_int = {token: index for index, token in int_to_note.items()}

    if seed_from_data:
        notes = midi_processor.extract_notes(data_dir)
        # Keep only tokens known to the vocabulary, then take a window.
        known = [note_to_int[t] for t in notes if t in note_to_int]
        if len(known) > sequence_length:
            start = random.randint(0, len(known) - sequence_length - 1)
            return known[start:start + sequence_length]
        print(
            "Warning: not enough seed data found; falling back to a random "
            "seed sequence."
        )

    # Random seed: pick `sequence_length` random valid token ids.
    return [random.randint(0, n_vocab - 1) for _ in range(sequence_length)]


def _decode_token_to_element(token, offset):
    """Turn a vocabulary token into a music21 Note or Chord at ``offset``.

    A token containing a '.' (or made of digits) is interpreted as a chord of
    normal-order pitch classes; otherwise it is treated as a single pitch name.
    """
    from music21 import chord, note

    # Chord tokens look like "0.4.7"; single notes look like "C4".
    if ("." in token) or token.isdigit():
        chord_notes = []
        for part in token.split("."):
            new_note = note.Note(int(part))
            chord_notes.append(new_note)
        new_element = chord.Chord(chord_notes)
    else:
        new_element = note.Note(token)

    new_element.offset = offset
    return new_element


def main(argv=None):
    """Entry point: load artifacts, generate tokens, and write a MIDI file."""
    args = parse_args(argv)

    # --- 1. Validate that the model and vocabulary exist ------------------ #
    if not os.path.isfile(args.model):
        print(
            "Model not found. Train first with: python train_model.py "
            "(expected model at '{}')".format(args.model)
        )
        sys.exit(1)

    if not os.path.isfile(args.vocab):
        print(
            "Vocabulary not found. Train first with: python train_model.py "
            "(expected vocab at '{}')".format(args.vocab)
        )
        sys.exit(1)

    # Heavy imports done here so the module imports cleanly without them.
    import numpy as np
    from tensorflow.keras.models import load_model
    from music21 import stream

    # --- 2. Load the model and vocabulary --------------------------------- #
    print("Loading model from: {}".format(args.model))
    try:
        model = load_model(args.model)
    except Exception as exc:  # noqa: BLE001 - present a friendly message
        print("Error: failed to load the model: {}".format(exc))
        sys.exit(1)

    int_to_note = midi_processor.load_vocab(args.vocab)
    n_vocab = len(int_to_note)
    if n_vocab == 0:
        print("Error: the vocabulary is empty. Re-run training.")
        sys.exit(1)

    # --- 3. Build the seed sequence --------------------------------------- #
    pattern = _build_seed(
        int_to_note, args.sequence_length, args.seed_from_data, args.data_dir
    )

    # --- 4. Iteratively predict the next token ---------------------------- #
    print("Generating {} notes...".format(args.length))
    generated_tokens = []
    for _ in range(args.length):
        # Shape input to (1, sequence_length, 1) and normalize like training.
        model_input = np.reshape(pattern, (1, len(pattern), 1))
        model_input = model_input / float(n_vocab)

        prediction = model.predict(model_input, verbose=0)
        index = int(np.argmax(prediction))
        generated_tokens.append(int_to_note[index])

        # Slide the window forward by one token.
        pattern.append(index)
        pattern = pattern[1:]

    # --- 5. Decode tokens into a music21 stream --------------------------- #
    output_notes = []
    offset = 0.0
    for token in generated_tokens:
        output_notes.append(_decode_token_to_element(token, offset))
        offset += OFFSET_STEP

    midi_stream = stream.Stream(output_notes)

    # --- 6. Write the MIDI file ------------------------------------------- #
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    midi_stream.write("midi", fp=args.out)
    print("Saved generated music to: {}".format(args.out))


if __name__ == "__main__":
    main()
