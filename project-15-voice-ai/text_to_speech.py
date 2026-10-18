
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

text = "Hello! I'm your AI assistant. How can I help you today?"

output_path = Path(__file__).parent / "outputs" / "speech.mp3"
output_path.parent.mkdir(parents=True, exist_ok=True)

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="coral",
    input=text,
    instructions="Speak naturally, in a friendly and professional tone."
) as response:
    response.stream_to_file(output_path)

print("Generated audio:", output_path)
