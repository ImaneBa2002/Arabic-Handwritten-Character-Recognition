
import numpy as np
from PIL import Image


def preprocess_handwritten_image(image):
    """
    Convert a handwritten character image to CNN input.

    Returns:
        float32 array with shape (32, 32, 1).
    """

    if image is None:
        raise ValueError("No image was provided.")

    # Composite transparent images over white.
    rgba = image.convert("RGBA")
    white_background = Image.new(
        "RGBA", rgba.size, (255, 255, 255, 255)
    )
    grayscale = Image.alpha_composite(
        white_background, rgba
    ).convert("L")

    pixels = np.asarray(
        grayscale, dtype=np.float32
    )

    # Estimate background intensity from the border.
    border = np.concatenate([
        pixels[0, :],
        pixels[-1, :],
        pixels[:, 0],
        pixels[:, -1]
    ])

    background = float(np.median(border))

    # Convert both polarities to light strokes on dark.
    if background >= 128:
        foreground = np.clip(
            background - pixels, 0, 255
        )
    else:
        foreground = np.clip(
            pixels - background, 0, 255
        )

    # Detect the handwritten character region.
    mask = foreground > 20

    if not np.any(mask):
        raise ValueError(
            "No visible handwritten character detected."
        )

    ys, xs = np.where(mask)

    x_min, x_max = xs.min(), xs.max() + 1
    y_min, y_max = ys.min(), ys.max() + 1

    cropped = foreground[
        y_min:y_max,
        x_min:x_max
    ]

    # Preserve the character aspect ratio.
    crop_height, crop_width = cropped.shape

    scale = 24 / max(crop_width, crop_height)

    new_width = max(
        1, int(round(crop_width * scale))
    )
    new_height = max(
        1, int(round(crop_height * scale))
    )

    cropped_image = Image.fromarray(
        cropped.astype(np.uint8)
    )

    resized = cropped_image.resize(
        (new_width, new_height),
        resample=Image.Resampling.LANCZOS
    )

    # Center the character on a 32x32 black canvas.
    canvas = Image.new("L", (32, 32), 0)

    left = (32 - new_width) // 2
    top = (32 - new_height) // 2

    canvas.paste(resized, (left, top))

    # Normalize exactly to the expected numerical range.
    processed = np.asarray(
        canvas, dtype=np.float32
    ) / 255.0

    return processed[..., np.newaxis]
