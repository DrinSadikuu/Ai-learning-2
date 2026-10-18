
import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.images.generate(
    model="gpt-image-1-mini",
    prompt=(
        "A modern smart home living room with large windows, "
        "minimalist furniture, warm lighting, and a small "
        "home assistant robot. Photorealistic style."
    ),
    size="1024x1024"
)

# The API returns the generated image as Base64
image_base64 = response.data[0].b64_json

# Decode Base64 back into image bytes
image_bytes = base64.b64decode(image_base64)

# Save the image
output_path = Path(__file__).parent / "outputs" / "generated_home.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

output_path.write_bytes(image_bytes)

print("Generated image saved to:", output_path)

with open(output_path, "rb") as image_file:
    edited_response = client.images.edit(
        model="gpt-image-1-mini",
        image=image_file,
        prompt=(
            "Keep the same living room layout and overall style. "
            "Change the sofa to blue and add a large indoor plant "
            "next to the window. Keep everything else as similar "
            "to the original as possible."
        ),
        size="1024x1024"
    )

# Decode the edited image
edited_bytes = base64.b64decode(
    edited_response.data[0].b64_json
)

# Save the edited image
edited_path = Path(__file__).parent / "outputs" / "edited_home.png"
edited_path.write_bytes(edited_bytes)

print("Edited image saved to:", edited_path)


