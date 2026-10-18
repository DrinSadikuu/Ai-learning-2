
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

project_dir = Path(__file__).parent
audio_path = project_dir / "audio" / "sample.m4a"
output_path = project_dir / "outputs" / "assistant_reply.mp3"

# 1. Convert your speech to text
with open(audio_path, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file
    )

question = transcription.text
print("You:", question)

# 2. Generate an AI response
response = client.responses.create(
    model="gpt-4.1-mini",
    input=question
)

answer = response.output_text
print("Assistant:", answer)

# 3. Convert the answer to speech
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="coral",
    input=answer
) as speech:
    speech.stream_to_file(output_path)

print("Audio response saved to:", output_path)
