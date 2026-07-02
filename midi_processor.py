"""
midi_processor.py
=================

MIDI parsing and dataset-preparation utilities for the Music Generation AI
project (CodeAlpha AI Internship - Task 3).

This module is responsible for:
  * Locating MIDI files in the dataset directory.
  * Extracting a flat sequence of note / chord tokens using ``music21``.
  * Building an integer vocabulary from those tokens.
  * Turning the token sequence into LSTM-ready training tensors.
  * Persisting / loading the vocabulary as JSON.

Heavy third-party imports (``music21`` and ``numpy``) are performed *inside*
the functions that need them so that this file can be imported (and
py_compiled) without those packages being installed.
"""

import json
import os

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Directory that holds the training MIDI files, resolved relative to this file
# so the pipeline works no matter what the current working directory is.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "midi")

# How many previous tokens the network looks at to predict the next token.
SEQUENCE_LENGTH = 32

# File extensions we treat as MIDI files.
_MIDI_EXTENSIONS = (".mid", ".midi")


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #

def list_midi_files(data_dir=DATA_DIR):
    """Return a sorted list of MIDI file paths found in ``data_dir``.

    Parameters
    ----------
    data_dir : str
        Directory to scan for ``.mid`` / ``.midi`` files.

    Returns
    -------
    list[str]
        Absolute (or as-given) paths to MIDI files, sorted alphabetically.
        An empty list is returned when the directory does not exist or has
        no MIDI files.
    """
    if not data_dir or not os.path.isdir(data_dir):
        return []

    files = []
    for name in os.listdir(data_dir):
        if name.lower().endswith(_MIDI_EXTENSIONS):
            files.append(os.path.join(data_dir, name))
    return sorted(files)


# --------------------------------------------------------------------------- #
# Note extraction
# --------------------------------------------------------------------------- #

def extract_notes(data_dir=DATA_DIR):
    """Parse every MIDI file in ``data_dir`` into a flat list of tokens.

    Each token is either:
      * a single note, represented as ``str(pitch)`` (e.g. ``"C4"``), or
      * a chord, represented as the '.'-joined ``normalOrder`` integers of the
        chord (e.g. ``"0.4.7"``).

    Parameters
    ----------
    data_dir : str
        Directory containing the MIDI files.

    Returns
    -------
    list[str]
        Flat list of note/chord tokens in the order they were encountered.
        Returns an empty list if there are no MIDI files (the caller is
        responsible for messaging the user in that case).
    """
    # Imported here so the module stays lightweight for py_compile / import.
    from music21 import chord, converter, instrument, note

    midi_files = list_midi_files(data_dir)
    if not midi_files:
        return []

    notes = []
    for path in midi_files:
        try:
            midi_stream = converter.parse(path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print("Warning: could not parse '{}': {}".format(path, exc))
            continue

        # Try to isolate a single instrument's part; fall back to a flat view.
        notes_to_parse = None
        try:
            parts = instrument.partitionByInstrument(midi_stream)
            if parts:  # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
        except Exception:  # noqa: BLE001 - partitioning is best-effort
            notes_to_parse = None

        if notes_to_parse is None:
            notes_to_parse = midi_stream.flatten().notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))

    return notes


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

def build_vocabulary(notes):
    """Build integer <-> token lookup tables from a list of tokens.

    Parameters
    ----------
    notes : list[str]
        Flat list of note/chord tokens.

    Returns
    -------
    tuple[dict, dict]
        ``(note_to_int, int_to_note)`` where ``note_to_int`` maps a token to
        its integer id and ``int_to_note`` is the reverse mapping. Tokens are
        assigned ids in sorted order for reproducibility.
    """
    unique_notes = sorted(set(notes))
    note_to_int = {token: index for index, token in enumerate(unique_notes)}
    int_to_note = {index: token for token, index in note_to_int.items()}
    return note_to_int, int_to_note


# --------------------------------------------------------------------------- #
# Sequence preparation
# --------------------------------------------------------------------------- #

def prepare_sequences(notes, note_to_int, sequence_length=SEQUENCE_LENGTH):
    """Convert tokens into normalized input windows and one-hot targets.

    The network input ``X`` has shape ``(n_patterns, sequence_length, 1)`` and
    is normalized to the ``[0, 1]`` range (dividing by ``n_vocab``). The target
    ``y`` is one-hot encoded with shape ``(n_patterns, n_vocab)`` so it can be
    trained with ``categorical_crossentropy``.

    Parameters
    ----------
    notes : list[str]
        Flat list of note/chord tokens.
    note_to_int : dict
        Token -> integer mapping from :func:`build_vocabulary`.
    sequence_length : int
        Number of tokens per input window.

    Returns
    -------
    tuple
        ``(X, y, n_vocab)`` ready to feed into a Keras LSTM model.

    Raises
    ------
    ValueError
        If there are not enough tokens to build a single training window.
    """
    import numpy as np

    n_vocab = len(note_to_int)
    if n_vocab == 0:
        raise ValueError("Vocabulary is empty; cannot prepare sequences.")

    if len(notes) <= sequence_length:
        raise ValueError(
            "Not enough notes ({}) for a sequence length of {}. "
            "Add more MIDI data or lower --sequence-length.".format(
                len(notes), sequence_length
            )
        )

    network_input = []
    network_output = []

    # Slide a window of `sequence_length` tokens across the corpus.
    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[token] for token in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)

    # Reshape into (samples, timesteps, features) and normalize to [0, 1].
    X = np.reshape(network_input, (n_patterns, sequence_length, 1))
    X = X / float(n_vocab)

    # One-hot encode the targets for categorical_crossentropy.
    y = np.zeros((n_patterns, n_vocab), dtype="float32")
    for row, target in enumerate(network_output):
        y[row, target] = 1.0

    return X, y, n_vocab


# --------------------------------------------------------------------------- #
# Vocabulary persistence
# --------------------------------------------------------------------------- #

def save_vocab(int_to_note, path):
    """Persist the ``int_to_note`` mapping to ``path`` as JSON.

    JSON object keys must be strings, so integer ids are stored as strings and
    converted back on load.

    Parameters
    ----------
    int_to_note : dict
        Integer id -> token mapping.
    path : str
        Destination JSON file path.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    serializable = {str(index): token for index, token in int_to_note.items()}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def load_vocab(path):
    """Load an ``int_to_note`` mapping previously written by :func:`save_vocab`.

    Parameters
    ----------
    path : str
        Path to the JSON vocabulary file.

    Returns
    -------
    dict
        Integer id -> token mapping (keys converted back to ``int``).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("Vocabulary file not found: {}".format(path))

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    return {int(index): token for index, token in raw.items()}
