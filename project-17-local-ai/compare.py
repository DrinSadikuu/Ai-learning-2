import time

from dotenv import load_dotenv
from ollama import chat
from openai import OpenAI

load_dotenv()
client = OpenAI()

question = "A shop sells a product for $120. It gives a 25% discount, then adds 10% tax to the discounted price. What is the final price? Explain your calculation."

# Local AI — Qwen
start = time.perf_counter()

local_response = chat(
    model="qwen2.5:1.5b",
    messages=[
        {"role": "user", "content": question}
    ]
)

local_time = time.perf_counter() - start

print("LOCAL AI — Qwen")
print("Answer:", local_response.message.content)
print("Time:", round(local_time, 2), "seconds")


# Hosted AI — OpenAI
start = time.perf_counter()

openai_response = client.responses.create(
    model="gpt-4.1-mini",
    input=question
)

openai_time = time.perf_counter() - start

print("\nOPENAI — GPT-4.1-mini")
print("Answer:", openai_response.output_text)
print("Time:", round(openai_time, 2), "seconds")