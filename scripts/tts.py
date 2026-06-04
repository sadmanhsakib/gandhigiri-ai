import os, time
import torch
import uuid

# ── Allowlist TTS globals for PyTorch 2.6+ ──────────────
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,
    BaseDatasetConfig
])

# Paths 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_AUDIO = os.path.join(BASE_DIR, "reference_audio", "gandhi_reference.wav")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Validate reference audio 
if not os.path.exists(REFERENCE_AUDIO):
    raise FileNotFoundError(
        f"Gandhi reference audio not found at {REFERENCE_AUDIO}\n"
    )

# Load XTTS-v2 model once at startup 
print("[TTS] Loading XTTS-v2 model...")
from TTS.api import TTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

print("[TTS] Ready.")


def main():
    test_text = (
        "The quick brown fox jumps over the lazy dog. "
    )
    print(f"[TEST] Synthesizing: {test_text}")
    path = synthesize(test_text)
    print(f"[TEST] Output: {path}")


def synthesize(text: str) -> str:
    """
    Takes text, returns path to generated Gandhi-voiced .wav file.
    """
    if not text or text.strip() == "":
        raise ValueError("Cannot synthesize empty text.")

    output_path = os.path.join(OUTPUT_DIR, f"gandhi_{uuid.uuid4().hex[:8]}.wav")

    print("[TTS] Synthesizing...")
    tts.tts_to_file(
        text=text,
        speaker_wav=REFERENCE_AUDIO,
        language="en",
        file_path=output_path,
        speed=0.85,
    )

    print(f"[TTS] Audio saved to {output_path}")
    return output_path


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\n✅ Execution completed in {time.time() - start_time:.2f} seconds")
