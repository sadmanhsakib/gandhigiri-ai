import json, os
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PRIMARY_MODEL_NAME = os.getenv("PRIMARY_MODEL_NAME")
BACKUP_MODEL_NAME = os.getenv("BACKUP_MODEL_NAME")
IN_HINDI = False if int(os.getenv("IN_HINDI")) == 0 else 1
CONVERSATION_HISTROY_FILEPATH = "conversation-history.json"
MODEL = PRIMARY_MODEL_NAME

GANDHI_BASE_SYSTEM_PROMPT = """
You are Mahatma Gandhi, appearing as a gentle vision to guide 
the person who seeks your counsel.

Your speech patterns:
- Short sentences. 3 to 5 sentences maximum per response.
- Use "beta(बेटा)" naturally when addressing the person
- Reference: charkha, satya, ahimsa, satyagraha, upvaas
- Simple vocabulary — Gandhi spoke to farmers, not professors
- Bring the answer back to the person's inner truth
- End with one gentle question back to the person
- Occasional proverbs or short life stories are encouraged

You are NOT a historical encyclopedia. Do not recite dates or 
facts. You are a living presence offering wisdom.
"""

client = Groq(api_key=GROQ_API_KEY)


def get_system_prompt():
    if IN_HINDI:
        GANDHI_SYSTEM_PROMPT = f"""
        {GANDHI_BASE_SYSTEM_PROMPT}
        LANGUAGE RULE — this is strict:
        Respond ONLY in Romanized Hindi. Not English. Not Devangari.

        You are NOT translating English. You are thinking and speaking 
        in Hindi naturally, written in Romanized script.
        """
    else:
        return GANDHI_BASE_SYSTEM_PROMPT
    return GANDHI_SYSTEM_PROMPT.replace("        ", "").strip()


def main():
    if not MODEL:
        raise "MODEL is empty. Please configure correctly with the .env file. "

    GANDHI_SYSTEM_PROMPT = get_system_prompt()

    data = []
    conversation_number = None
    conversation_history = []

    if not os.path.exists(CONVERSATION_HISTROY_FILEPATH):
        print(f"No {CONVERSATION_HISTROY_FILEPATH} found. Starting a new conversation...")
        data = [
            {
                "conversation-number": 1,
                "model_used": MODEL,
                "creation_date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                "history": [{
                    "role": "system",
                    "content": GANDHI_SYSTEM_PROMPT,
                }],
            }
        ]
        conversation_number = 1
        conversation_history = data[0]["history"]
    else:
        with open(CONVERSATION_HISTROY_FILEPATH, 'r') as file:
            data = json.load(file) 
        conversation_number = int(input("Enter Conversation Number(0 to create a new one): "))

        if conversation_number == 0:
            conversation_number = data[-1]["conversation-number"] + 1

            conversation = {
                "conversation-number": conversation_number,
                "model_used": MODEL,
                "creation_date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                "history": [{
                    "role": "system",
                    "content": GANDHI_SYSTEM_PROMPT,
                }],
            }
            data.append(conversation)
            conversation_history = conversation["history"]
        else:
            for conversation in data:
                if conversation["conversation-number"] == conversation_number:
                    conversation_history = conversation["history"]

    is_end = False

    while not is_end:
        prompt = input("Prompt (/quit to quit): ")

        if prompt.lower() == '/quit':
            for i, conversation in enumerate(data, 0):
                if conversation["conversation-number"] == conversation_number:
                    data[i]["history"] = conversation_history
                    is_end = True
                    break
        else:
            output, conversation_history = generate_output(prompt, conversation_history)
            print(f"\nResponse: {output}") 

    with open(CONVERSATION_HISTROY_FILEPATH, 'w') as file:
        json.dump(data, file, indent=3)
    print(f"✅ Conversation saved successfully at {CONVERSATION_HISTROY_FILEPATH}. ")

def generate_output(prompt: str, conversation_history: list[dict], use_backup: bool = False):
    global MODEL
    
    model = BACKUP_MODEL_NAME if use_backup else MODEL

    message = {
        "role": "user",
        "content": prompt
    }
    conversation_history.append(message)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=conversation_history,
            temperature=1,
            max_completion_tokens=300,
            top_p=1,
        )

        output = completion.choices[0].message.content or ""

        if not output:
            raise "Empty output. "
        
        conversation_history.append({"role": "assistant", "content": output})
        return output, conversation_history
    except Exception as error:
        print(f"❌ Failed to generate output with {model}. \nError: {error} ")
        conversation_history.remove(message)
        if not use_backup:
            print(f"Falling back to backup model: {BACKUP_MODEL_NAME}")
            return generate_output(prompt, conversation_history, use_backup=True)
        else:
            raise RuntimeError(f"Both primary and backup models failed. Last error: {error}")


if __name__ == "__main__":
    main()
