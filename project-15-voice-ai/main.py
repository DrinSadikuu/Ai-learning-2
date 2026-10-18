
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from .env
load_dotenv()
client = OpenAI()

# Locate our recorded audio
audio_path = Path(__file__).parent / "audio" / "sample.m4a"

# Send audio to the speech recognition model
with open(audio_path, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file
    )

# Print the recognized speech
print("Transcription:", transcription.text)
