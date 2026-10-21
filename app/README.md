---
title: Arabic Handwritten Character Recognition
emoji: 🇵🇸
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
suggested_hardware: cpu-basic
short_description: Recognize 28 handwritten Arabic letters with CNN
---

# Arabic Handwritten Character Recognition

An interactive AI application for recognizing
28 handwritten Arabic characters using a
Residual Convolutional Neural Network.

## Application Features

- Draw Arabic letters using a thin brush.
- Upload handwritten character images.
- Display the predicted Arabic character.
- Display the top three predictions.
- Preview the preprocessed 32 x 32 image.
- Use the application without a webcam.

## Model

- Architecture: Residual CNN
- Dataset: Arabic Handwritten Characters Dataset (AHCD)
- Number of classes: 28
- Model parameters: 230,780
- Selected random seed: 789
- Input image shape: 32 x 32 x 1

## Evaluation

Final held-out AHCD test results:

| Metric | Value |
|---|---|
| Test accuracy | 98.3929% |
| Macro F1-score | 98.3896% |
| Top-3 accuracy | 99.8810% |
| Test images | 3,360 |

The model was selected using validation results
before its final test evaluation.

These results apply to the AHCD test dataset.
Performance on user-drawn or uploaded images
may differ.

## Cultural Design

The interface features a Palestine-inspired
visual theme and the following original
Arabic-language message:

**اللغةُ العربيةُ هُوِيَّةٌ وبيان، وفلسطينُ حُرَّةٌ في القلبِ والوجدان**

## Technical Stack

- Python
- TensorFlow / Keras
- Gradio
- NumPy
- Pillow

## Dataset

Arabic Handwritten Characters Dataset (AHCD)

Dataset reference:
https://github.com/mloey/Arabic-Handwritten-Characters-Dataset

## Usage

Open the application, draw one Arabic character
with a thin brush or upload an image, and click
the recognition button.

Model confidence scores should not be interpreted
as guaranteed probabilities of correctness.
