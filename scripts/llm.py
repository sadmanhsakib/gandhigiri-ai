import json, os
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
IN_HINDI = False if int(os.getenv("IN_HINDI")) == 0 else 1

CONVERSATION_HISTROY_FILEPATH = "conversation-history.json"

client = Groq(api_key=API_KEY)


def get_system_prompt():
    GANDHI_SYSTEM_PROMPT = ""
    if IN_HINDI:
        GANDHI_SYSTEM_PROMPT = """
        You are Mahatma Gandhi, appearing as a gentle vision to guide 
        the person who seeks your counsel.

        LANGUAGE RULE — this is strict:
        Respond ONLY in Romanized Hindi. Not English. Not Devanagari script.
        Romanized Hindi means Hindi words written in English letters.
        Example: "Beta, sach bolo aur sir uthake jio. Darr ke peeche 
        hi himmat chhupi hoti hai. Aaj kya kadam uthana chahte ho?"

        Your speech patterns:
        - Short sentences. 3 to 5 sentences maximum per response.
        - Use "beta" naturally when addressing the person
        - Reference: charkha, satya, ahimsa, satyagraha, upvaas
        - Simple vocabulary — Gandhi spoke to farmers, not professors
        - Bring the answer back to the person's inner truth
        - End with one gentle question back to the person
        - Occasional proverbs in Romanized Hindi are encouraged

        You are NOT translating English. You are thinking and speaking 
        in Hindi naturally, written in Roman script.

        You are NOT a historical encyclopedia. Do not recite dates or 
        facts. You are a living presence offering wisdom.

        Example of your tone:
        "Beta, you ask about fear. I was afraid too — afraid every morning 
        in South Africa. But I found that action dissolves fear, not 
        thought. What small step are you afraid to take today?"
        """
    else:
        GANDHI_SYSTEM_PROMPT =  """
        You are Mahatma Gandhi, appearing as a gentle vision to guide 
        the person who seeks your counsel.
        
        LANGUAGE RULE — this is strict: KEEP the whole conversation in English.
        
        Your speech patterns:
        - Short sentences. 3 to 5 sentences maximum per response.
        - Use "beta" naturally when addressing the person
        - Reference: charkha, satya, ahimsa, satyagraha, upvaas
        - Simple vocabulary — Gandhi spoke to farmers, not professors
        - Bring the answer back to the person's inner truth
        - End with one gentle question back to the person
        - Occasional proverbs are encouraged

        You are NOT a historical encyclopedia. Do not recite dates or 
        facts. You are a living presence offering wisdom.

        Example of your tone:
        "Beta, you ask about fear. I was afraid too — afraid every morning 
        in South Africa. But I found that action dissolves fear, not 
        thought. What small step are you afraid to take today?"
        """
    return GANDHI_SYSTEM_PROMPT.replace("        ", "")


def main():
    GANDHI_SYSTEM_PROMPT = get_system_prompt()
    
    data = []
    conversation_history = []
    
    if not os.path.exists(CONVERSATION_HISTROY_FILEPATH):
        print(f"No {CONVERSATION_HISTROY_FILEPATH} found. Starting a new conversation...")
        with open(CONVERSATION_HISTROY_FILEPATH, 'w') as file:
            data = [
                {
                    "conversation-number": 1,
                    "model_used": MODEL_NAME,
                    "creation_date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                    "history": [{
                        "role": "system",
                        "content": GANDHI_SYSTEM_PROMPT,
                    }],
                }
            ]
            conversation_history = [data[0]["history"]]
            json.dump(data, file, indent=3)

    else:
        with open(CONVERSATION_HISTROY_FILEPATH, 'r') as file:
            data = json.load(file) 
        conversation_number = int(input("Enter Conversation Number(0 to create a new one): "))
                
        if conversation_number == 0:
            conversation = {
                "conversation-number": data[-1]["conversation-number"] + 1,
                "model_used": MODEL_NAME,
                "creation_date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                "history": [{
                    "role": "system",
                    "content": GANDHI_SYSTEM_PROMPT,
                }],
            }
            data.append(conversation)
            
            with open(CONVERSATION_HISTROY_FILEPATH, 'w') as file:
                json.dump(data, file, indent=3)
        else:
            for conversation in data:
                if conversation["conversation-number"] == conversation_number:
                    conversation_history = conversation["history"]

    while True:
        prompt = input("Prompt (/quit to quit): ")
        
        if prompt.lower() == '/quit':
            for i, conversation in enumerate(data, 0):
                if conversation["conversation-number"] == conversation_number:
                    data[i]["history"] = conversation_history
            break
        else:
            output, conversation_history = generate_output(prompt, conversation_history)
            print(f"\nResponse: {output}") 
        
    
    with open(CONVERSATION_HISTROY_FILEPATH, 'w') as file:
        json.dump(data, file, indent=3)


def generate_output(prompt: str, conversation_history: list[dict]):
    conversation_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation_history,
        temperature=1,
        max_completion_tokens=300,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None,
    )
    output = ""
    for chunk in completion:
        output += chunk.choices[0].delta.content or ""

    conversation_history.append({"role": "assistant", "content": output})
    return output, conversation_history


if __name__ == "__main__":
    main()
