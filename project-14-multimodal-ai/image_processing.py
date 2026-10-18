
from pathlib import Path

from PIL import Image, ImageOps


def process_image(
    image_path: Path,
    output_dir: Path,
    max_size: int = 1024,
    quality: int = 85
) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_processed.jpg"

    with Image.open(image_path) as image:

        # Apply orientation information from the image
        image = ImageOps.exif_transpose(image)

        # Convert to RGB (JPEG does not support transparency)
        if image.mode in ("RGBA", "P"):
            background = Image.new("RGB", image.size, "white")
            rgba_image = image.convert("RGBA")
            background.paste(rgba_image, mask=rgba_image.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")

        # Resize while preserving proportions
        image.thumbnail((max_size, max_size))

        # Save as JPEG
        image.save(
            output_path,
            format="JPEG",
            quality=quality
        )

    return output_path


# Test the pipeline
project_dir = Path(__file__).parent

input_image = project_dir / "images" / "sample.png"
output_directory = project_dir / "outputs"

result_path = process_image(
    image_path=input_image,
    output_dir=output_directory
)

print("Processed image:", result_path)
print("Original size:", round(input_image.stat().st_size / 1024, 2), "KB")
print("Processed size:", round(result_path.stat().st_size / 1024, 2), "KB")
