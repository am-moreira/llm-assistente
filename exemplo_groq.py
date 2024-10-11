import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explique a importância de modelos de linguagem rápidos",
        }
    ],
    model="llama3-8b-8192",
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="", flush=True)