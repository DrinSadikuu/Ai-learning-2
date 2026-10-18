
import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


# Convert an image file into a Base64 data URL
def encode_image(image_name: str) -> str:
    image_path = Path(__file__).parent / "images" / image_name

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:image/png;base64,{base64_image}"


# Prepare both images
image1 = encode_image("sample.png")
image2 = encode_image("sample2.png")


# Send both images to OpenAI
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Compare Image 1 and Image 2. "
                        "Describe what is similar and what is different. "
                        "Focus on vehicles, shops, buildings, and people. "
                        "Do not assume these images show the exact same street."
                    )
                },
                {
                    "type": "input_image",
                    "image_url": image1
                },
                {
                    "type": "input_image",
                    "image_url": image2
                }
            ]
        }
    ]
)

print(response.output_text)
