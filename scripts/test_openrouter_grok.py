import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_grok():
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("Error: OPEN_ROUTER_API_KEY not found in .env")
        return

    # OpenRouter Configuration
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Model: Grok 4.1 Fast
    # Note: "Grok 4.1 Fast" might not be the exact OpenRouter ID yet.
    # Adjust MODEL_NAME if necessary. Common Grok IDs: 'x-ai/grok-beta', 'x-ai/grok-2'
    MODEL_NAME = "x-ai/grok-4.1-fast" # Placeholder for latest/fastest. 
    # If specifically 'grok-4.1-fast' exists in future, change to that.
    
    print(f"Testing connectivity with model: {MODEL_NAME}")
    print("(Note: If specific 'Grok 4.1' ID isn't valid yet, this might fallback or fail)")

    try:
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Spotify Agentic RAG Test",
            },
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Please reply with 'Grok is ready!' if you receive this."}
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
    test_grok()
