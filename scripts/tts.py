"""
tts.py
------
Converts LLM-generated text (English or romanized Hindi) into spoken audio.
This audio is the "raw material" that voice_conversion.py (RVC) will later
re-skin into Gandhi's voice.

Pipeline:
    text -> (if romanized Hindi: transliterate to Devanagari)
         -> edge-tts -> WAV file
"""

import asyncio
import os
import edge_tts
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ---------------------------------------------------------------------------
# Voice configuration
# ---------------------------------------------------------------------------
# Pick voices that sound calm/measured -- closer to Gandhi's cadence,
# which helps RVC produce a more natural final result.
ENGLISH_VOICE = "en-IN-PrabhatNeural"  # Indian-English male voice
HINDI_VOICE = "hi-IN-MadhurNeural"  # Hindi male voice

# Rate/pitch tweaks: Gandhi's speech is slow and deliberate.
# edge-tts supports SSML-like prosody via these params.
SPEECH_RATE = "-10%"
SPEECH_PITCH = "-5Hz"


def main():
    samples = [
        ("The path you are walking on, son, will one day get you into trouble.", False),
        ("जिस राह पे तुम चल रहे हो बेटा एक दिन बुरी तरह फसोगे।", True),
    ]
    
    sample = samples[1]
    output_path = text_to_speech(text=sample[0], output_in_romanized_hindi=sample[1])
    print(f"✅ File exported successfully to {output_path}")

def romanized_to_devanagari(text: str) -> str:
    """
    Converts romanized Hindi (e.g., "Tumhe sach ka raasta chunna hoga")
    into Devanagari script so edge-tts's Hindi voice pronounces it correctly.

    Uses ITRANS scheme as input since that's the closest match to how
    people casually type romanized Hindi (phonetic spelling).
    """
    return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)


async def _synthesize(text: str, voice: str, output_path: str):
    """Core TTS function"""
    communicate = edge_tts.Communicate(
        text,
        voice,
        rate=SPEECH_RATE,
        pitch=SPEECH_PITCH,
    )
    try:
        await communicate.save(output_path)
    except FileNotFoundError:
        os.makedirs("/".join(output_path.split("/")[:-1]), exist_ok=False)
        await communicate.save(output_path)
        

def text_to_speech(
    text: str, output_in_romanized_hindi: bool = False, output_path: str = "audio_output/output.wav"
) -> str:
    """
    Main entry point. Prepares text, and generates audio.

    Args:
        text: LLM output, either English or romanized Hindi.
        output_in_romanized_hindi: LLM output language.
        output_path: where to save the generated audio file.

    Returns:
        Path to the generated audio file (str), for downstream RVC step.
    """
    if not text or not text.strip():
        raise ValueError("text_to_speech received empty text")
    if output_in_romanized_hindi:
        voice = HINDI_VOICE
        final_text = romanized_to_devanagari(text)
    else:
        voice = ENGLISH_VOICE
        final_text = text
    
    asyncio.run(_synthesize(final_text, voice, output_path))
    return output_path


if __name__ == "__main__":
    main()
