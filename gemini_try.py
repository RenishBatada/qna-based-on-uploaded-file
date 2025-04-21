from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=gemini_api_key)

result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents="What is the meaning of life?")

print(result.embeddings)