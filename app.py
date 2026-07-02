"""
app.py
======

Streamlit demo UI for the Music Generation AI project
(CodeAlpha AI Internship - Task 3).

What this app does:
  * Shows how many MIDI files are in ``data/midi/``.
  * Lets you create synthetic demo MIDI files for a quick smoke test.
  * Shows the (terminal-only) training command, because training is heavy.
  * If a trained model exists, lets you generate a MIDI file and download it.

The app never claims a model is trained when it is not; training must be run
from a terminal. Heavy work (music21 / TensorFlow) is invoked via helper
imports and a subprocess so the UI stays responsive and importable.
"""

import os
import subprocess
import sys

import streamlit as st

import midi_processor
from train_model import NO_DATA_MESSAGE


# --------------------------------------------------------------------------- #
# Paths (resolved relative to this file so the app is location-independent).
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "music_lstm.keras")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "models", "vocab.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "generated_music.mid")


def count_midi_files():
    """Return the number of MIDI files currently in the dataset directory."""
    return len(midi_processor.list_midi_files(midi_processor.DATA_DIR))


def render_dataset_status():
    """Show dataset status and the demo-creation button."""
    st.subheader("1. Dataset status")

    midi_count = count_midi_files()
    st.metric("MIDI files in data/midi/", midi_count)

    if midi_count == 0:
        # Exact guidance message shared with the training script.
        st.warning(NO_DATA_MESSAGE)

    if st.button("Create demo MIDI (smoke test)"):
        with st.spinner("Creating synthetic demo MIDI files..."):
            try:
                # Imported lazily; needs music21 to be installed.
                from scripts.create_demo_midi import create_demo_midi

                written = create_demo_midi()
            except ImportError:
                st.error(
                    "Could not import the demo generator. Make sure "
                    "'music21' is installed: pip install -r requirements.txt"
                )
                return
            except Exception as exc:  # noqa: BLE001 - surface a friendly note
                st.error("Failed to create demo MIDI files: {}".format(exc))
                return

        st.success(
            "Created {} demo MIDI file(s). These are SYNTHETIC and only for "
            "smoke-testing the pipeline.".format(len(written))
        )
        for path in written:
            st.write("- `{}`".format(os.path.relpath(path, PROJECT_ROOT)))
        # Rerun so the metric above refreshes with the new count.
        st.rerun()


def render_training_section():
    """Explain training and show the terminal command."""
    st.subheader("2. Train the model (runs in a terminal)")
    st.write(
        "Training is compute-heavy, so it is **not** run inside this app. "
        "Open a terminal in the project folder and run:"
    )
    st.code("python train_model.py --epochs 5", language="bash")
    st.caption(
        "Epochs are configurable. The model is saved to "
        "`models/music_lstm.keras` and the vocabulary to `models/vocab.json`."
    )


def render_generation_section():
    """Show the generation UI when a trained model is present."""
    st.subheader("3. Generate music")

    model_ready = os.path.isfile(MODEL_PATH) and os.path.isfile(VOCAB_PATH)

    if not model_ready:
        st.info(
            "No trained model found yet. Train first with "
            "`python train_model.py --epochs 5`, then reload this page."
        )
        return

    st.success("A trained model was found. You can generate music now.")
    length = st.slider(
        "Number of notes to generate", min_value=50, max_value=500, value=200,
        step=10,
    )

    if st.button("Generate music"):
        with st.spinner("Generating music... this may take a moment."):
            # Run generation as a subprocess so heavy TensorFlow work is
            # isolated from the Streamlit process.
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "generate_music.py"),
                "--length", str(length),
                "--model", MODEL_PATH,
                "--vocab", VOCAB_PATH,
                "--out", OUTPUT_PATH,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=PROJECT_ROOT
            )

        if result.returncode != 0:
            st.error("Generation failed. Details below:")
            st.code(result.stderr or result.stdout or "No output.")
            return

        if not os.path.isfile(OUTPUT_PATH):
            st.error(
                "Generation reported success but no output file was found at "
                "`outputs/generated_music.mid`."
            )
            return

        st.success("Music generated successfully.")
        with open(OUTPUT_PATH, "rb") as handle:
            st.download_button(
                label="Download generated_music.mid",
                data=handle.read(),
                file_name="generated_music.mid",
                mime="audio/midi",
            )


def main():
    """Render the full Streamlit page."""
    st.set_page_config(page_title="Music Generation AI", page_icon="music")
    st.title("Music Generation with AI")
    st.write(
        "An LSTM-based music generator built for the CodeAlpha AI Internship "
        "(Task 3). Use the steps below to create demo data, train, and "
        "generate a new MIDI file."
    )

    render_dataset_status()
    st.divider()
    render_training_section()
    st.divider()
    render_generation_section()


if __name__ == "__main__":
    main()
