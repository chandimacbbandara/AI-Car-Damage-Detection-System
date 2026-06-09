import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import gradio as gr
from PIL import Image

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 1. Load Custom YOLOv8 Model
weights_path = 'best.pt'
if not os.path.exists(weights_path):
    print("Warning: best.pt not found. Running mock/empty model mode for setup.")
    damage_model = None
    class_names = {i: f"damage_type_{i}" for i in range(7)}
else:
    print("Loading custom YOLOv8 model from best.pt...")
    damage_model = YOLO(weights_path)
    class_names = damage_model.names
    print(f"Model loaded successfully. Classes: {class_names}")

# 2. Setup Vehicle Verification Model (ResNet-18)
print("Loading ResNet-18 for identity verification...")
try:
    resnet_weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=resnet_weights)
except Exception as e:
    print(f"Error loading ResNet-18 weights: {e}. Loading uninitialized ResNet-18.")
    resnet = models.resnet18()
feature_extractor = nn.Sequential(*(list(resnet.children())[:-1]))
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

# Preprocessing for ResNet embedding extraction
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(img_rgb):
    """Extract features from an image using ResNet-18."""
    img_t = preprocess(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = feature_extractor(img_t)
        emb = emb.squeeze()
        emb = emb / (emb.norm() + 1e-8)
    return emb

def verify_single_view(img_b, img_a):
    """Calculate cosine similarity between before and after image embeddings."""
    emb_b = get_image_embedding(img_b)
    emb_a = get_image_embedding(img_a)
    similarity = torch.dot(emb_b, emb_a).item()
    return similarity

def align_images(img_before, img_after):
    """Align the 'after' image to the 'before' image using ORB and Homography."""
    # Convert images to grayscale (input is RGB)
    gray_before = cv2.cvtColor(img_before, cv2.COLOR_RGB2GRAY)
    gray_after = cv2.cvtColor(img_after, cv2.COLOR_RGB2GRAY)

    # Detect ORB features
    orb = cv2.ORB_create(nfeatures=8000)
    kp1, des1 = orb.detectAndCompute(gray_before, None)
    kp2, des2 = orb.detectAndCompute(gray_after, None)

    if des1 is None or des2 is None:
        return img_after

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Require at least 15 good matches
    good_matches = max(int(len(matches) * 0.15), 15)
    if len(matches) < 15:
        print("⚠️ Poor structural alignment detected. Keeping original coordinates.")
        return img_after

    points_before = np.zeros((good_matches, 2), dtype=np.float32)
    points_after = np.zeros((good_matches, 2), dtype=np.float32)

    for i, match in enumerate(matches[:good_matches]):
        points_before[i, :] = kp1[match.queryIdx].pt
        points_after[i, :] = kp2[match.trainIdx].pt

    h, mask = cv2.findHomography(points_after, points_before, cv2.RANSAC, 5.0)

    # Safety Check: If the warping matrix is too extreme, don't use it
    if h is None or np.abs(np.linalg.det(h)) < 0.6 or np.abs(np.linalg.det(h)) > 1.4:
        print("⚠️ Skewed alignment matrix rejected. Keeping image raw.")
        return img_after

    height, width, channels = img_before.shape
    return cv2.warpPerspective(img_after, h, (width, height))

def calculate_iou(box_a, box_b):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = float(box_a_area + box_b_area - inter_area)
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def process_inspection(
    front_b, rear_b, left_b, right_b,
    front_a, rear_a, left_a, right_a,
    conf_threshold, iou_threshold, sim_threshold
):
    views = ["Front", "Rear", "Left", "Right"]
    before_imgs = [front_b, rear_b, left_b, right_b]
    after_imgs = [front_a, rear_a, left_a, right_a]

    # Validate that we have at least one valid before/after pair
    provided_views = []
    for i, (b, a) in enumerate(zip(before_imgs, after_imgs)):
        if b is not None and a is not None:
            provided_views.append(i)

    if not provided_views:
        status_html = """
        <div class="result-banner error">
            <h3>❌ Error: Incomplete Inputs</h3>
            <p>Please upload at least one view with both 'Before' and 'After' images to perform the inspection.</p>
        </div>
        """
        return status_html, None, None, None, None, "### Summary Report\n\nNo views were uploaded for analysis."

    total_similarity = 0.0
    view_sim_scores = {}
    annotated_after_imgs = [None, None, None, None]
    damage_details = []
    
    # Process each uploaded view
    for idx in provided_views:
        view_name = views[idx]
        img_b = before_imgs[idx]
        img_a = after_imgs[idx]

        # Standardize sizes if necessary, but ORB handles different sizes
        img_b = np.array(img_b)
        img_a = np.array(img_a)

        # Align AFTER image to BEFORE image
        aligned_a = align_images(img_b, img_a)

        # Verify identity using ResNet
        sim_score = verify_single_view(img_b, aligned_a)
        total_similarity += sim_score
        view_sim_scores[view_name] = sim_score

        # If model is not loaded (mock mode), skip detection
        if damage_model is None:
            annotated_after_imgs[idx] = aligned_a
            damage_details.append(f"**{view_name} View**: Model weights not found, skipping detection. Identity Match Score: {sim_score:.2f}")
            continue

        # Run custom YOLOv8 model
        # We use a very low confidence threshold for the baseline (Before) image (e.g., 0.15)
        # to ensure any pre-existing parts are detected and filtered out.
        before_conf = min(0.15, conf_threshold)
        results_before = damage_model(img_b, conf=before_conf, verbose=False)[0]
        results_after = damage_model(aligned_a, conf=conf_threshold, verbose=False)[0]

        boxes_before = results_before.boxes.xyxy.cpu().numpy()
        boxes_after = results_after.boxes.xyxy.cpu().numpy()
        classes_after = results_after.boxes.cls.cpu().numpy()
        conf_after = results_after.boxes.conf.cpu().numpy()

        new_damages_found = 0
        display_img = aligned_a.copy()
        view_damages = []

        for j, box_after in enumerate(boxes_after):
            is_pre_existing = False

            # Check if this box overlaps with any box in BEFORE
            for box_before in boxes_before:
                if calculate_iou(box_after, box_before) > iou_threshold:
                    is_pre_existing = True
                    break

            # If it doesn't overlap, it is a new damage!
            if not is_pre_existing:
                new_damages_found += 1
                cls_id = int(classes_after[j])
                cls_name = class_names[cls_id]
                conf = conf_after[j]

                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, box_after)
                label = f"NEW: {cls_name} ({conf:.2f})"

                # Draw red box
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (239, 68, 68), 3)
                # Draw text background
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y_label_start = max(y1, label_size[1] + 10)
                cv2.rectangle(
                    display_img, 
                    (x1, y_label_start - label_size[1] - 6), 
                    (x1 + label_size[0] + 6, y_label_start + base_line - 4), 
                    (239, 68, 68), 
                    cv2.FILLED
                )
                # Draw white text
                cv2.putText(
                    display_img, 
                    label, 
                    (x1 + 3, y_label_start - 3), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.55, 
                    (255, 255, 255), 
                    2, 
                    cv2.LINE_AA
                )

                view_damages.append(f"**{cls_name}** (Confidence: {conf:.1%})")

        annotated_after_imgs[idx] = display_img

        if view_damages:
            damage_details.append(f"🔴 **{view_name} View**: Detected {new_damages_found} new damage(s): {', '.join(view_damages)}")
        else:
            damage_details.append(f"🟢 **{view_name} View**: No new damages detected.")

    # Calculate average identity score
    avg_similarity = total_similarity / len(provided_views)

    # Construct HTML Status Banner
    warning_msg = ""
    if len(provided_views) < 4:
        warning_msg = f"<p style='color: #b45309; margin: 5px 0 0 0; font-size: 0.95rem; font-weight: 500;'>⚠️ Running in partial view mode ({len(provided_views)}/4 sides uploaded).</p>"

    if avg_similarity < sim_threshold:
        status_html = f"""
        <div class="result-banner error">
            <h3>❌ IDENTITY MATCH REFUSED</h3>
            <p>The system has determined that the uploaded images do not match the same vehicle or the photo perspectives differ significantly.</p>
            <p class="score">Average Identity Score: <strong>{avg_similarity:.3f}</strong> (Required Threshold: {sim_threshold})</p>
            {warning_msg}
        </div>
        """
        summary_md = f"### Inspection Refused\n\n**Average Identity Score**: {avg_similarity:.3f}\n\n" + "\n".join(damage_details)
    else:
        status_html = f"""
        <div class="result-banner success">
            <h3>✅ VEHICLE IDENTITY MATCHED</h3>
            <p>Verification successful. Below are the aligned 'After' views highlighting any newly localized damages.</p>
            <p class="score">Average Identity Score: <strong>{avg_similarity:.3f}</strong> (Required Threshold: {sim_threshold})</p>
            {warning_msg}
        </div>
        """
        summary_md = f"### Inspection Report\n\n**Average Identity Score**: {avg_similarity:.3f}\n\n" + "\n".join(damage_details)

    return (
        status_html,
        annotated_after_imgs[0],
        annotated_after_imgs[1],
        annotated_after_imgs[2],
        annotated_after_imgs[3],
        summary_md
    )

# Premium stylesheet
css = """
/* Force light theme colors for all states (even dark mode) */
:root, .dark, [data-theme="dark"], html, body, .gradio-container {
    --body-background-fill: #ffffff !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #e2e8f0 !important;
    --border-color-primary: #e2e8f0 !important;
    --border-color-secondary: #f1f5f9 !important;
    --text-color: #0f172a !important;
    --body-text-color: #0f172a !important;
    --block-label-text-color: #475569 !important;
    --block-title-text-color: #1e293b !important;

    /* Input variables for text and slider number inputs */
    --input-background-fill: #f8fafc !important;
    --input-border-color: #e2e8f0 !important;
    --input-text-color: #0f172a !important;
    --input-background-fill-focus: #ffffff !important;
    --input-border-color-focus: #3b82f6 !important;
    --input-text-color-focus: #0f172a !important;
    --input-placeholder-color: #94a3b8 !important;
    
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #0f172a !important;
}

/* Ensure high visibility of text elements in all conditions */
h1, h2, h3, h4, h5, h6, p, span, li, label, .prose, .markdown, .gr-form, .gr-box, .gr-input, .gr-button {
    color: #0f172a !important;
}

/* Explicit fallback styling for input elements (including the slider number boxes) */
input, textarea, select, .gr-input, .gr-text-input, .gr-number-input {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    border: 1px solid #e2e8f0 !important;
}
input:focus, textarea:focus, select:focus {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-color: #3b82f6 !important;
    outline: none !important;
}

/* Keep warning and error states correctly colored */
.result-banner.success h3, .result-banner.success p, .result-banner.success strong {
    color: #166534 !important;
}
.result-banner.error h3, .result-banner.error p, .result-banner.error strong {
    color: #991b1b !important;
}

.container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 1rem !important;
}
.header {
    text-align: center;
    padding: 2.5rem 1.5rem;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    border-radius: 16px;
    box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    color: white;
}
.header h1 {
    font-size: 2.75rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.05em;
    color: #ffffff !important;
}
.header p {
    font-size: 1.15rem;
    color: #e0f2fe !important;
    margin-top: 0.5rem;
    font-weight: 400;
}
.result-banner {
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    border-left-width: 6px;
    border-left-style: solid;
}
.result-banner.success {
    background-color: #f0fdf4 !important;
    border-left-color: #22c55e !important;
}
.result-banner.error {
    background-color: #fef2f2 !important;
    border-left-color: #ef4444 !important;
}
.result-banner h3 {
    margin-top: 0;
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.result-banner p {
    margin: 0.25rem 0;
    font-size: 1.05rem;
}
.result-banner .score {
    font-size: 1.1rem;
    margin-top: 0.75rem;
    padding-top: 0.5rem;
    border-top: 1px solid rgba(0,0,0,0.05);
}
.run-btn {
    background: linear-gradient(90deg, #3b82f6, #2563eb) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.3) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.run-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.4) !important;
}
.run-btn:active {
    transform: translateY(1px) !important;
}
"""

with gr.Blocks(title="AutoInspect AI - Vehicle Damage Inspection System") as demo:
    gr.HTML("""
    <div class="header">
        <h1>🚘 AutoInspect AI</h1>
        <p>State-of-the-Art Vehicle Identity Verification & Damage Detection</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            conf_threshold = gr.Slider(minimum=0.1, maximum=1.0, value=0.1, step=0.05, label="YOLOv8 Detection Confidence")
            iou_threshold = gr.Slider(minimum=0.1, maximum=1.0, value=0.1, step=0.05, label="Damage Match Overlap (IoU)")
            sim_threshold = gr.Slider(minimum=0.5, maximum=1.0, value=0.5, step=0.01, label="Identity Match Threshold (ResNet-18)")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📅 Before State (Reference Pictures)")
            front_b = gr.Image(label="Front View (BEFORE)", type="numpy", height=450)
            rear_b = gr.Image(label="Rear View (BEFORE)", type="numpy", height=450)
            left_b = gr.Image(label="Left View (BEFORE)", type="numpy", height=450)
            right_b = gr.Image(label="Right View (BEFORE)", type="numpy", height=450)

        with gr.Column():
            gr.Markdown("### 🔍 After State (Verification Pictures)")
            front_a = gr.Image(label="Front View (AFTER)", type="numpy", height=450)
            rear_a = gr.Image(label="Rear View (AFTER)", type="numpy", height=450)
            left_a = gr.Image(label="Left View (AFTER)", type="numpy", height=450)
            right_a = gr.Image(label="Right View (AFTER)", type="numpy", height=450)

    gr.HTML("<br>")
    run_btn = gr.Button("🔍 Run Inspection & Verification", elem_classes="run-btn", variant="primary")
    gr.HTML("<br>")

    gr.Markdown("### 📊 Inspection Results")
    status_html = gr.HTML(value="<div style='text-align: center; padding: 2rem; color: #64748b; font-size: 1.1rem; border: 2px dashed #e2e8f0; border-radius: 12px;'>Upload images and click 'Run Inspection' to view results.</div>")

    with gr.Tabs():
        with gr.TabItem("Front View Result"):
            front_res = gr.Image(label="Front View Inspection", height=500)
        with gr.TabItem("Rear View Result"):
            rear_res = gr.Image(label="Rear View Inspection", height=500)
        with gr.TabItem("Left View Result"):
            left_res = gr.Image(label="Left View Inspection", height=500)
        with gr.TabItem("Right View Result"):
            right_res = gr.Image(label="Right View Inspection", height=500)

    gr.HTML("<br>")
    summary_text = gr.Markdown("### Summary Report\nNo inspection has been run yet.")

    run_btn.click(
        fn=process_inspection,
        inputs=[
            front_b, rear_b, left_b, right_b,
            front_a, rear_a, left_a, right_a,
            conf_threshold, iou_threshold, sim_threshold
        ],
        outputs=[
            status_html,
            front_res, rear_res, left_res, right_res,
            summary_text
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=css)
