import gradio as gr
import cv2
import numpy as np
import PIL.Image as Image
from PIL import ImageDraw, ImageFont
from ultralytics import YOLO

# Load available models
models = {
    
    "YOLOv8n Baseline": YOLO("yolov8n_baseline.pt"),
    "YOLOv8n lr1e4_do01": YOLO("yolov8n_lr1e-4_do0.1.pt"),
    "YOLOv8n lr5e3_do01": YOLO("yolov8n_lr5e-3_do0.1.pt"),
    "YOLOv8n lr1e4_do03": YOLO("yolov8n_lr1e-4_do0.3.pt"),
    "YOLOv8n lr5e3_do03": YOLO("yolov8n_lr5e-3_do0.3.pt"),
    "YOLOv8n Preprocessed": YOLO("yolov8n_preprocessed.pt"),
}

# Enhance function (noise reduction using Gaussian Blur)
def noiseReduction_image(image):
    # Convert PIL Image to OpenCV format
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur for noise reduction
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    # Return a 3-channel image for compatibility
    return cv2.merge([denoised, denoised, denoised])

def overlay_boxes_on_original(original_img, results, model):
    """Draw bounding boxes from processed image results on the original image."""
    draw = ImageDraw.Draw(original_img)
    width, height = original_img.size

    # Scale factors for bounding box thickness and text size
    box_thickness = max(2, int(min(width, height) / 300))
    font_size = max(10, int(min(width, height) / 50))

    # Try to load a custom font, fallback to default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            # Draw bounding box (blue color)
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=box_thickness)

            # Add label and confidence
            label = f"{model.names[int(cls)]} {conf:.2f}"
            
            # Use textbbox to get the bounding box for the text
            text_bbox = draw.textbbox((x1, max(0, y1 - font_size - 2)), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Draw blue background for the text
            text_bg = [x1, max(0, y1 - text_height - 2), x1 + text_width, y1]
            draw.rectangle(text_bg, fill="blue")

            # Draw label text in white on the blue background
            draw.text((x1, max(0, y1 - text_height - 2)), label, fill="white", font=font)

    return original_img

def predict_image(img, model_name, conf_threshold, iou_threshold):
    """Predicts objects in an image using a YOLO model with adjustable confidence and IOU thresholds."""
    # Get the selected model
    model = models[model_name]

    # Keep the original image for output
    original_img = img.copy()

    # Conditionally preprocess the image if the model is YOLOv8n Preprocessed
    if model_name == "YOLOv8n Preprocessed":
        preprocessed_img = noiseReduction_image(img)
    else:
        preprocessed_img = img  # Use the original image without preprocessing

    # Run the model on the preprocessed image
    results = model.predict(
        source=np.array(preprocessed_img),
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
    )

    # Overlay results on the original image
    annotated_img = overlay_boxes_on_original(original_img, results, model)

    return annotated_img

# Create Gradio Interface
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(choices=list(models.keys()), value="YOLOv8n Baseline", label="Select Model"),
        gr.Slider(minimum=0, maximum=1, value=0.5, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.3, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Aplikasi Deteksi Pothole",
    description="1301210233 - Kahfi Rizky Firmansyah <br> 1301213251 - Gandhi Risyad Abimanyu",
)

if __name__ == "__main__":
    iface.launch()
