# CodeAlpha Music Generation AI

## 1. Project Title
**CodeAlpha Music Generation AI** — an LSTM-based deep learning system that
learns musical patterns from MIDI files and generates brand-new melodies.

## 2. Objective
Build an Artificial Intelligence model that can **generate original music**.
The system parses a corpus of MIDI files into a sequence of note/chord tokens,
trains a recurrent neural network (LSTM) to predict the next token from a short
history, and then uses the trained model to compose a new MIDI file one note at
a time.

## 3. Features
- Parses `.mid` / `.midi` files into note and chord tokens using **music21**.
- Builds a reproducible integer vocabulary from the training corpus.
- Prepares normalized, LSTM-ready training sequences with **NumPy**.
- Trains a stacked **LSTM** network (TensorFlow / Keras) with configurable
  epochs, sequence length, and batch size.
- Generates new music by iteratively predicting the next note and writing a
  MIDI file.
- Ships a **synthetic demo MIDI generator** so the whole pipeline can be
  smoke-tested without downloading a dataset.
- Includes a **Streamlit** UI to check dataset status, create demo data,
  view the training command, and generate/download music.
- Clear, friendly error messages when data, models, or dependencies are
  missing — no raw tracebacks and no fake success messages.

## 4. Tech Stack
- **Language:** Python 3
- **Deep Learning:** TensorFlow / Keras (LSTM)
- **Music/MIDI:** music21
- **Numerics:** NumPy
- **UI:** Streamlit
- **Standard library:** argparse, json, os, subprocess

## 5. Folder Structure
```
CodeAlpha_Music_Generation_AI/
├── app.py                     # Streamlit demo UI
├── midi_processor.py          # MIDI parsing, vocab, sequence prep, vocab I/O
├── train_model.py             # Trains the LSTM model (CLI)
├── generate_music.py          # Generates a new MIDI file (CLI)
├── requirements.txt           # Project dependencies
├── README.md                  # This file
├── scripts/
│   ├── __init__.py
│   └── create_demo_midi.py    # Creates synthetic demo MIDI files
├── data/
│   └── midi/
│       └── .gitkeep           # Place your real MIDI dataset here
├── models/
│   └── .gitkeep               # Trained model + vocab.json are written here
└── outputs/
    └── .gitkeep               # generated_music.mid is written here
```

## 6. Installation Steps
1. Ensure **Python 3.9+** is installed.
2. (Recommended) create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 7. Run Command
Full pipeline, in order (run from the project root):
```bash
# 1) Create synthetic demo MIDI files for a quick smoke test (optional)
python scripts/create_demo_midi.py

# 2) Train the LSTM model (configurable epochs)
python train_model.py --epochs 5

# 3) Generate a new MIDI file from the trained model
python generate_music.py --length 200

# 4) Launch the Streamlit demo UI
streamlit run app.py
```

## 8. Input and Output
- **Input:** MIDI files (`.mid` / `.midi`) placed in `data/midi/`.
- **Model artifacts:** trained model saved to `models/music_lstm.keras` and
  vocabulary saved to `models/vocab.json`.
- **Output:** a generated MIDI file at `outputs/generated_music.mid`, which you
  can play in any MIDI-capable player or DAW, or download from the Streamlit UI.

## 9. Screenshots
_Add screenshots here after running._

## 10. Known Limitations
- The included demo MIDI files are **synthetic** (simple scales/arpeggios) and
  exist **only** to smoke-test the pipeline — they will not produce musically
  interesting results.
- Meaningful output requires a **real MIDI dataset** in `data/midi/` and many
  more training epochs than the default of 5.
- The default `--epochs 5` is intentionally small so the pipeline runs quickly;
  quality improves substantially with more epochs and more data.
- Training is compute-heavy and is intended to run in a terminal, not inside the
  Streamlit app.
- Generation uses a simple greedy (argmax) strategy, which can be repetitive.

## 11. Future Improvements
- Add temperature-based / top-k sampling for more varied, less repetitive music.
- Model note **durations** and **rests** in addition to pitch.
- Support multi-instrument / polyphonic generation.
- Add model checkpointing, a validation split, and early stopping.
- Render generated MIDI to audio (WAV/MP3) for in-browser playback.

## 12. CodeAlpha Submission Note
This project was built for the **CodeAlpha Artificial Intelligence Internship —
Task 3: Music Generation with AI**. It demonstrates an end-to-end deep learning
pipeline (data processing → LSTM training → music generation) with a Streamlit
demo interface.

> Note: The bundled demo MIDI files are synthetic and only for smoke-testing.
> Real training requires a real MIDI dataset placed in `data/midi/`.
