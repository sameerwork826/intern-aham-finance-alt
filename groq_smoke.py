import os
from llm_client import LLMClient, sysmsg, usermsg

def main():
    print("LLM_PROVIDER:", os.getenv("LLM_PROVIDER"))
    print("GROQ_MODEL:", os.getenv("GROQ_MODEL"))
    print("GROQ_API_KEY set:", bool(os.getenv("GROQ_API_KEY")))
    try:
        client = LLMClient()
        print("Provider:", client.provider)
        resp = client.chat([sysmsg("You are a test."), usermsg("Say OK")], max_tokens=4)
        print("Response:", resp)
    except Exception as e:
        import traceback
        print("ERROR:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()


