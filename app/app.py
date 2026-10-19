
# ============================================================
# ARABIC HANDWRITTEN CHARACTER RECOGNITION
# PALESTINE-INSPIRED GRADIO APPLICATION
# ============================================================

import gradio as gr
import tensorflow as tf
import numpy as np
import json
import html
import sys

from pathlib import Path
from PIL import Image

# ============================================================
# 1. LOAD MODEL AND PREPROCESSING MODULE
# ============================================================

APP_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(APP_DIR))

from preprocessing import preprocess_handwritten_image

MODEL_PATH = APP_DIR / "arabic_residual_cnn.keras"
LABELS_PATH = APP_DIR / "class_labels.json"

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_labels = json.load(f)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

assert len(class_labels) == 28
assert model.count_params() == 230780

# ============================================================
# 2. PALESTINE-INSPIRED DESIGN
# ============================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}

.palestine-ribbon {
    position: relative;
    width: 100%;
    height: 17px;
    border-radius: 12px;
    overflow: hidden;
    background: linear-gradient(
        to bottom,
        #171717 0%,
        #171717 33.33%,
        #ffffff 33.33%,
        #ffffff 66.66%,
        #007a3d 66.66%,
        #007a3d 100%
    );
    margin-bottom: 25px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.10);
}

.palestine-ribbon::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 64px;
    height: 100%;
    background: #ce1126;
    clip-path: polygon(0 0, 100% 50%, 0 100%);
}

.hero {
    text-align: center;
    padding: 16px 18px 26px;
}

.hero-title {
    color: #153b31;
    font-size: 33px;
    font-weight: 800;
    line-height: 1.3;
    margin: 0 0 10px;
}

.hero-subtitle {
    color: #64746f;
    font-size: 16px;
    line-height: 1.7;
}

.guide-box {
    text-align: center;
    background: #f2f8f4;
    color: #264c3e;
    padding: 14px;
    border: 1px solid #d7e8dc;
    border-radius: 14px;
    margin-bottom: 12px;
}

.section-heading {
    color: #174e3d;
    font-size: 21px;
    font-weight: 800;
    margin: 20px 0 12px;
}

.prediction-card {
    background: linear-gradient(
        135deg, #ffffff, #f3f9f5
    );
    border: 1px solid #d8e8de;
    border-radius: 20px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(15,60,40,0.06);
}

.predicted-letter {
    font-family: 'Amiri', 'Noto Naskh Arabic',
                 'Traditional Arabic', serif;
    font-size: 94px;
    font-weight: 700;
    color: #087849;
    line-height: 1.5;
    direction: rtl;
}

.heritage-card {
    margin: 22px 0 12px;
    padding: 30px 24px;
    border-radius: 22px;
    background: linear-gradient(
        135deg, #fdfaf7, #f2f8f4
    );
    border: 1px solid #e7e2d9;
    text-align: center;
    box-shadow: 0 8px 26px rgba(30,45,35,0.07);
    animation: gentle-appear 0.55s ease-out;
}

.heritage-quote {
    font-family: 'Amiri', 'Noto Naskh Arabic',
                 'Traditional Arabic', serif;
    font-size: 29px;
    font-weight: 700;
    color: #214e3e;
    line-height: 2.1;
    direction: rtl;
}

.heritage-caption {
    color: #7b867e;
    font-size: 12px;
    margin-top: 14px;
    letter-spacing: 0.5px;
}

@keyframes gentle-appear {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

button.pal-action {
    background: #087849 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    min-height: 46px !important;
}

button.pal-action:hover {
    background: #065d39 !important;
}

.app-footer {
    text-align: center;
    color: #7d8b84;
    font-size: 12px;
    line-height: 1.8;
    padding: 18px 8px;
}

@media (max-width: 600px) {
    .hero-title {
        font-size: 25px;
    }

    .heritage-quote {
        font-size: 23px;
    }

    .predicted-letter {
        font-size: 76px;
    }
}
"""

# ============================================================
# 3. BUILD PREDICTION DISPLAY
# ============================================================

def make_prediction_card(letter, confidence):

    safe_letter = html.escape(letter)

    return f"""
    <div class="prediction-card">

        <div style="color:#687b72;font-size:15px;">
            Recognized Arabic Character
        </div>

        <div class="predicted-letter">
            {safe_letter}
        </div>

        <div style="color:#33483e;font-size:16px;">
            Model confidence score:
            <strong>{confidence:.2%}</strong>
        </div>

        <p style="color:#829187;font-size:12px;">
            Confidence is not a guarantee of correctness.
        </p>

    </div>
    """


def make_heritage_message():

    return """
    <div class="heritage-card">

        <div style="font-size:24px;margin-bottom:10px;">
            ✨ 🇵🇸 ✨
        </div>

        <div class="heritage-quote" lang="ar" dir="rtl">
            اللغةُ العربيةُ هُوِيَّةٌ وبيان،
            وفلسطينُ حُرَّةٌ في القلبِ والوجدان
        </div>

        <div class="heritage-caption">
            A LETTER • A LANGUAGE • A HERITAGE
        </div>

    </div>
    """

# ============================================================
# 4. CHARACTER RECOGNITION
# ============================================================

def recognize_character(image):

    if image is None:
        return (
            "<div class='prediction-card'>"
            "Please draw or upload a character first."
            "</div>",
            None,
            {},
            ""
        )

    try:
        # Apply the deployment preprocessing pipeline
        processed = preprocess_handwritten_image(image)

        input_batch = np.expand_dims(
            processed,
            axis=0
        )

        # Predict using the trained Residual CNN
        probabilities = model(
            input_batch,
            training=False
        ).numpy()[0]

        top_indices = np.argsort(
            probabilities
        )[-3:][::-1]

        predicted_id = int(top_indices[0])

        predicted_letter = class_labels[
            str(predicted_id)
        ]

        confidence = float(
            probabilities[predicted_id]
        )

        # Generate display elements
        prediction_html = make_prediction_card(
            predicted_letter,
            confidence
        )

        top3 = {
            class_labels[str(int(idx))]:
                float(probabilities[idx])
            for idx in top_indices
        }

        preview = np.clip(
            processed[:, :, 0] * 255,
            0,
            255
        ).astype(np.uint8)

        heritage_html = make_heritage_message()

        return (
            prediction_html,
            preview,
            top3,
            heritage_html
        )

    except Exception as exc:

        return (
            "<div class='prediction-card'>"
            "Unable to process image: "
            + html.escape(str(exc))
            + "</div>",
            None,
            {},
            ""
        )


def recognize_drawing(editor_value):

    if editor_value is None:
        return recognize_character(None)

    composite = editor_value.get("composite")

    return recognize_character(composite)

# ============================================================
# 5. BUILD GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    css=CUSTOM_CSS,
    title="Arabic Handwritten Character Recognition",
    theme=gr.themes.Soft(
        primary_hue="green",
        neutral_hue="slate"
    )
) as demo:

    # Header
    gr.HTML("""
    <div class="palestine-ribbon"></div>

    <div class="hero">

        <div style="font-size:42px;">✍️</div>

        <h1 class="hero-title">
            Arabic Handwritten Character Recognition
        </h1>

        <p class="hero-subtitle">
            Discover the beauty of Arabic handwriting
            through Artificial Intelligence.
            <br>
            Powered by a Residual Convolutional
            Neural Network.
        </p>

    </div>
    """)

    # Instructions
    gr.HTML("""
    <div class="guide-box">
        Draw one Arabic letter with a thin brush
        or upload a handwritten image.
        <br>
        Explore how AI recognizes
        the 28 Arabic characters.
    </div>
    """)

    # Inputs
    with gr.Tabs():

        with gr.Tab("✍️ Draw a Letter"):

            drawing_input = gr.ImageEditor(
                value=Image.new(
                    "RGB", (256, 256), "white"
                ),
                type="pil",
                image_mode="RGB",
                sources=[],
                canvas_size=(256, 256),
                fixed_canvas=True,
                brush=gr.Brush(
                    colors=["#000000"],
                    default_color="#000000",
                    color_mode="fixed",
                    default_size=6
                ),
                layers=False,
                label="Draw One Arabic Character"
            )

            draw_button = gr.Button(
                "Recognize Drawing",
                variant="primary",
                elem_classes=["pal-action"]
            )

        with gr.Tab("📁 Upload an Image"):

            upload_input = gr.Image(
                type="pil",
                sources=["upload", "clipboard"],
                label="Upload an Arabic Handwritten Character",
                height=280
            )

            gr.Markdown(
                "Use a clear, centered handwritten "
                "character on a simple background."
            )

            upload_button = gr.Button(
                "Recognize Uploaded Image",
                variant="primary",
                elem_classes=["pal-action"]
            )

    # Results
    gr.HTML("""
    <div class="section-heading">
        ✨ Recognition Results
    </div>
    """)

    prediction_html = gr.HTML(
        "<div class='prediction-card'>"
        "Your recognized character will appear here."
        "</div>"
    )

    with gr.Row():

        processed_preview = gr.Image(
            label="CNN Input — 32 × 32",
            interactive=False,
            image_mode="L",
            height=190
        )

        top3_output = gr.Label(
            label="Top 3 Predictions",
            num_top_classes=3
        )

    heritage_message = gr.HTML(value="")

    # Connect prediction events
    draw_button.click(
        fn=recognize_drawing,
        inputs=drawing_input,
        outputs=[
            prediction_html,
            processed_preview,
            top3_output,
            heritage_message
        ]
    )

    upload_button.click(
        fn=recognize_character,
        inputs=upload_input,
        outputs=[
            prediction_html,
            processed_preview,
            top3_output,
            heritage_message
        ]
    )

    # Footer
    gr.HTML("""
    <div class="app-footer">
        Residual CNN · AHCD Dataset ·
        28 Arabic Characters

        <br>
        Educational AI Prototype

        <br>
        Recognition performance on user drawings
        may differ from the AHCD test results.
    </div>
    """)

# ============================================================
# 6. LAUNCH IN STANDALONE ENVIRONMENTS
# ============================================================

if __name__ == "__main__":
    demo.launch()
