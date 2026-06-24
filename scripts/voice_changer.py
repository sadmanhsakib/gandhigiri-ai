"""
voice_changer.py
----------------
Converts TTS-generated audio into Gandhi's voice by calling Applio's
inference engine via subprocess — using Applio's own isolated venv,
so no pip dependency conflicts with your project's Python 3.12 env.

Pipeline position:
    tts.py (any voice) → voice_changer.py (Gandhi's voice) → playback
"""

import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent.parent.resolve()  # project root
APPLIO_DIR    = BASE_DIR / "Applio"
APPLIO_PYTHON = Path(f"{BASE_DIR}/.venv/Scripts/python.exe")
APPLIO_INFER  = APPLIO_DIR / "core.py"

MODEL_PATH    = BASE_DIR / "models" / "voice_changer_200e_6000s.pth"
INDEX_PATH    = BASE_DIR / "models" / "gandhi_model.index"

OUTPUT_DIR    = BASE_DIR / "audio_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------
PITCH_SHIFT          = 0       # semitones. Try -2 to -4 if too high-pitched.
INDEX_INFLUENCE      = 0.66    # 0.0 - 1.0. Higher = more Gandhi timbre.
CONSONANT_PROTECTION = 0.33    # 0.0 - 0.5. Protects consonants from distortion.
PITCH_METHOD         = "rmvpe" # best quality: rmvpe, fcpe, or hybrid[rmvpe+fcpe]
CLEAN_AUDIO          = "True"  # noise reduction on output — recommended for speech
CLEAN_STRENGTH       = 0.5     # 0.0 - 1.0. Higher = stronger cleaning.
VOLUME_ENVELOPE      = 0.25    # 0.0 - 1.0. Blends output volume envelope.


def _validate():
    """Check all required files/folders exist before attempting inference."""
    if not APPLIO_DIR.exists():
        raise FileNotFoundError(
            f"Applio folder not found at: {APPLIO_DIR}\n"
            f"Make sure Applio is cloned inside your project folder."
        )
    if not APPLIO_PYTHON.exists():
        raise FileNotFoundError(
            f"Python executable not found at: {APPLIO_PYTHON}\n"
            f"Update APPLIO_PYTHON path at the top of this file."
        )
    if not APPLIO_INFER.exists():
        raise FileNotFoundError(
            f"Applio core.py not found at: {APPLIO_INFER}\n"
            f"Your Applio installation may be incomplete."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Download G_2333333.pth from Google Drive into models/"
        )
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index file not found: {INDEX_PATH}\n"
            f"Download added_*.index from Google Drive into models/"
        )


def _run_inference(
    input_audio_path: Path,
    output_filename: str,
    pitch_shift: int = PITCH_SHIFT,
) -> Path:
    """
    Calls Applio's core.py infer subcommand via subprocess.
    Must be run with cwd=APPLIO_DIR so core.py can find its relative assets.
    Returns the path to the converted output file.
    """
    output_path = OUTPUT_DIR / output_filename

    cmd = [
        str(APPLIO_PYTHON),
        str(APPLIO_INFER),
        "infer",
        "--pitch",           str(pitch_shift),
        "--index_rate",      str(INDEX_INFLUENCE),
        "--volume_envelope", str(VOLUME_ENVELOPE),
        "--protect",         str(CONSONANT_PROTECTION),
        "--f0_method",       PITCH_METHOD,
        "--input_path",      str(input_audio_path),
        "--output_path",     str(output_path),
        "--pth_path",        str(MODEL_PATH),
        "--index_path",      str(INDEX_PATH),
        "--clean_audio",     CLEAN_AUDIO,
        "--clean_strength",  str(CLEAN_STRENGTH),
        "--export_format",   "WAV",
    ]

    print(f"[voice_changer] Converting: {input_audio_path.name} -> {output_filename}")

    result = subprocess.run(
        cmd,
        cwd=str(APPLIO_DIR),  # critical: core.py needs to run from its own directory
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[voice_changer] Applio stderr:\n{result.stderr}")
        raise RuntimeError(
            f"Applio inference failed with return code {result.returncode}.\n"
            f"Check the error above."
        )

    if not output_path.exists():
        raise RuntimeError(
            f"Inference completed but output file not found: {output_path}\n"
            f"Applio stdout: {result.stdout}"
        )

    print(f"[voice_changer] Done. Gandhi audio saved to: {output_path}")
    return output_path


def convert_to_gandhi_voice(
    input_audio_path: str,
    output_filename: str = "gandhi_output.wav",
) -> str:
    """
    Converts a TTS-generated audio file into Gandhi's voice.

    Args:
        input_audio_path: path to the TTS output WAV/MP3 (from tts.py)
        output_filename:  filename for the Gandhi-voiced output

    Returns:
        Path to the Gandhi-voiced output audio file as a string.
    """
    input_path = Path(input_audio_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    _validate()
    output = _run_inference(input_path, output_filename, PITCH_SHIFT)
    return str(output)


def tune_pitch(
    input_audio_path: str,
    shifts: list[int] = [-4, -2, 0, 2, 4],
):
    """
    Generates multiple versions at different pitch shifts so you can
    A/B test which sounds most like Gandhi. Run once after training,
    pick the best, then hardcode that value in PITCH_SHIFT above.
    """
    input_path = Path(input_audio_path).resolve()
    _validate()

    print("[voice_changer] Generating pitch variants for tuning...")
    for shift in shifts:
        filename = f"pitch_test_{shift:+d}.wav"
        _run_inference(input_path, filename, pitch_shift=shift)
        print(f"  Shift {shift:+d} semitones -> {OUTPUT_DIR / filename}")

    print(
        f"\n[voice_changer] Listen to each file in audio_output/ and pick "
        f"the best sounding one. Then set PITCH_SHIFT to that value."
    )


if __name__ == "__main__":
    input_file = "output/output.wav"
    tune_pitch(input_file)
