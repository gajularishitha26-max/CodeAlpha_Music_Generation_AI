"""
train_model.py
==============

Train an LSTM neural network on a corpus of MIDI files so it learns to predict
the next musical token (note or chord) given a short sequence of previous
tokens. Part of the CodeAlpha AI Internship - Task 3 (Music Generation with AI).

Usage
-----
    python train_model.py --epochs 5

The heavy TensorFlow import lives inside ``main()`` so this file imports and
py_compiles cleanly without TensorFlow being installed.
"""

import argparse
import os
import sys

import midi_processor


# The exact guidance message shown when no MIDI data is available. Kept as a
# module-level constant so the Streamlit app can display the identical text.
NO_DATA_MESSAGE = (
    "Please add MIDI files into data/midi/ or run "
    "scripts/create_demo_midi.py for a smoke-test demo."
)


def parse_args(argv=None):
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train an LSTM model for AI music generation."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=32,
        help="Length of each input sequence (default: 32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64).",
    )
    parser.add_argument(
        "--data-dir",
        default=midi_processor.DATA_DIR,
        help="Directory containing training MIDI files.",
    )
    parser.add_argument(
        "--model-out",
        default=os.path.join("models", "music_lstm.keras"),
        help="Where to save the trained Keras model.",
    )
    parser.add_argument(
        "--vocab-out",
        default=os.path.join("models", "vocab.json"),
        help="Where to save the vocabulary JSON.",
    )
    return parser.parse_args(argv)


def build_model(n_vocab, sequence_length):
    """Construct and compile the LSTM model.

    Architecture: two stacked LSTM layers with dropout, a dense hidden layer,
    and a softmax output over the vocabulary. Compiled with the Adam optimizer
    and categorical cross-entropy loss.

    Parameters
    ----------
    n_vocab : int
        Size of the token vocabulary (number of output classes).
    sequence_length : int
        Number of timesteps in each input window.

    Returns
    -------
    keras.Model
        A compiled, ready-to-train model.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential(name="music_lstm")
    # Input shape is (timesteps, features=1); return sequences to stack LSTMs.
    model.add(
        LSTM(
            256,
            input_shape=(sequence_length, 1),
            return_sequences=True,
        )
    )
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def main(argv=None):
    """Entry point: extract data, build vocab, train, and save artifacts."""
    args = parse_args(argv)

    # --- 1. Extract note/chord tokens from the MIDI dataset ---------------- #
    print("Scanning for MIDI files in: {}".format(args.data_dir))
    notes = midi_processor.extract_notes(args.data_dir)

    if not notes:
        # Exact message required by the project specification.
        print(NO_DATA_MESSAGE)
        sys.exit(1)

    print("Extracted {} note/chord tokens.".format(len(notes)))

    # --- 2. Build the vocabulary ------------------------------------------ #
    note_to_int, int_to_note = midi_processor.build_vocabulary(notes)
    print("Vocabulary size: {} unique tokens.".format(len(note_to_int)))

    # --- 3. Prepare training tensors -------------------------------------- #
    try:
        X, y, n_vocab = midi_processor.prepare_sequences(
            notes, note_to_int, args.sequence_length
        )
    except ValueError as exc:
        print("Error: {}".format(exc))
        sys.exit(1)

    print(
        "Prepared {} training sequences of length {}.".format(
            X.shape[0], args.sequence_length
        )
    )

    # --- 4. Build and train the model ------------------------------------- #
    print("Building LSTM model...")
    model = build_model(n_vocab, args.sequence_length)
    model.summary()

    print("Training for {} epoch(s)...".format(args.epochs))
    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size)

    # --- 5. Save the model and vocabulary --------------------------------- #
    model_dir = os.path.dirname(os.path.abspath(args.model_out))
    os.makedirs(model_dir, exist_ok=True)
    model.save(args.model_out)
    print("Saved trained model to: {}".format(args.model_out))

    midi_processor.save_vocab(int_to_note, args.vocab_out)
    print("Saved vocabulary to: {}".format(args.vocab_out))

    print("Training complete.")


if __name__ == "__main__":
    main()
