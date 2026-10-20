from ollama import chat

response = chat(
    model="qwen2.5:1.5b",
    options={
      "temperature": 1
    },
    messages=[
        {
            "role": "system",
            "content": "You are a Python teacher. Explain everything using very simple English."
        },
        {
            "role": "user",
            "content": "What is a Python decorator?"
        }
    ]
)

print(response.message.content)