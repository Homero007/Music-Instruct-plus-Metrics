import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("ERROR: HF_TOKEN no está definido. Por favor, asegúrate de tener un archivo .env en este directorio con tu token: HF_TOKEN='tu_token_aqui'")
    sys.exit(1)

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
)

try:
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct:nscale",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
    )
    print("Respuesta de la API:")
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"Error al conectar con la API de Hugging Face: {e}")
