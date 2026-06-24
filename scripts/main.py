import os
import json
from pathlib import Path
from playsound3 import playsound
import llm
import tts
import voice_changer

BASE_DIR = Path(__file__).parent.parent.resolve()


def main():
    if not os.path.exists(BASE_DIR / ".env"):
        raise "⚠️ .env file not found."

    data, conversation_number, conversation_history = llm.initialize()

    is_end = False
    actual_audio_path = ""
    
    while not is_end:
        prompt = input("Prompt (/quit to quit): ")

        if prompt.lower() == '/quit':
            for i, conversation in enumerate(data, 0):
                if conversation["conversation-number"] == conversation_number:
                    data[i]["history"] = conversation_history
                    is_end = True
                    break
        else:
            if prompt == "" and actual_audio_path != "":
                playsound(actual_audio_path)
                continue
            output, conversation_history = llm.generate_output(prompt, conversation_history)
            print(f"\nResponse: {output}")
            print("(Generating voice...)")
            tmp_audio_path = tts.text_to_speech(text=output)
            actual_audio_path = voice_changer.convert_to_gandhi_voice(tmp_audio_path)
            playsound(actual_audio_path)

    with open(llm.CONVERSATION_HISTROY_FILEPATH, 'w') as file:
        json.dump(data, file, indent=3)
    print(f"✅ Conversation saved successfully at {llm.CONVERSATION_HISTROY_FILEPATH}. ")


if __name__ == "__main__":
    main()
