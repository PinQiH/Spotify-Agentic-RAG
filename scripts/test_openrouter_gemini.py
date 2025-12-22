import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini():
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("Error: OPEN_ROUTER_API_KEY not found in .env")
        return

    # OpenRouter Configuration
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Model: Gemini 2.0 Flash
    # Note: Model IDs on OpenRouter can change. 
    # Common IDs: 'google/gemini-2.0-flash-exp:free', 'google/gemini-2.0-flash-exp'
    MODEL_NAME = "google/gemini-2.0-flash-001"

    print(f"Testing connectivity with model: {MODEL_NAME}")
    
    try:
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:8501", # Optional, for OpenRouter rankings
                "X-Title": "Spotify Agentic RAG Test",
            },
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Please reply with 'Gemini 2.0 Flash is online!' if you receive this."}
            ],
            temperature=0.7,
        )
        
        reply = response.choices[0].message.content
        print("\nSuccess! Response received:")
        print("-" * 30)
        print(reply)
        print("-" * 30)

    except Exception as e:
        print(f"\nError calling OpenRouter: {e}")

if __name__ == "__main__":
    test_gemini()
