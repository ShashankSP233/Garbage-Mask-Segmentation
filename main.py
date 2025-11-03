import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from segment_anything import sam_model_registry, SamPredictor
import torch
import os
import glob

# ---- SETTINGS ----
image_folder = "test"      # folder with input images
checkpoint_path = "sam_vit_b_01ec64.pth"  # update path
model_type = "vit_b"
output_folder = "output_masks"
csv_path = "mask_results.csv"

os.makedirs(output_folder, exist_ok=True)

# ---- DEVICE ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---- Load SAM model ----
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamPredictor(sam)

# ---- Preprocessing (CLAHE + Edge Enhancement) ----
def preprocess_image(image_rgb):
    """Apply CLAHE + Edge Enhancement to boost segmentation quality."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Convert to LAB and apply CLAHE
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Edge detection
    gray = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Blend CLAHE + Edges
    boosted = cv2.addWeighted(image_clahe, 0.8, edges_colored, 0.2, 0)

    return cv2.cvtColor(boosted, cv2.COLOR_BGR2RGB)

# ---- Mask cleaning ----
def clean_mask(mask, img_shape):
    h, w = img_shape[:2]
    k = max(3, min(h, w) // 200)
    kernel = np.ones((k, k), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned

# ---- CSV results ----
results = []

# ---- Process images ----
image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
              glob.glob(os.path.join(image_folder, "*.png"))

for img_path in image_paths:
    print(f"\nProcessing: {img_path}")
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print(f" Skipping (could not load): {img_path}")
        continue
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ---- Apply preprocessing ----
    preprocessed_rgb = preprocess_image(image_rgb)

    # Resize for ROI selection
    display_height = 720
    scale = display_height / preprocessed_rgb.shape[0]
    resized_image = cv2.resize(preprocessed_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # ---- User selects ROI ----
    bbox = cv2.selectROI(f"Select ROI - {os.path.basename(img_path)}", resized_image,
                         fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if bbox == (0, 0, 0, 0):
        print("No ROI selected, skipping...")
        continue

    # Rescale bbox to original
    x_scaled, y_scaled, w_scaled, h_scaled = bbox
    x = int(x_scaled / scale)
    y = int(y_scaled / scale)
    w = int(w_scaled / scale)
    h = int(h_scaled / scale)
    input_box = np.array([x, y, x + w, y + h])

# ---- Predict with SAM ----
predictor.set_image(preprocessed_rgb)
masks, scores, _ = predictor.predict(
    box=input_box[None, :],
    multimask_output=True
)

# Clean masks
cleaned_masks = [clean_mask(m, preprocessed_rgb.shape) for m in masks]

# Pick best
best_idx = np.argmax(scores)
best_score = scores[best_idx]

# Save masks
base_name = os.path.splitext(os.path.basename(img_path))[0]
for i, mask in enumerate(cleaned_masks):
    rgba = np.dstack((preprocessed_rgb, (mask * 255).astype(np.uint8)))
    save_path = os.path.join(output_folder, f"{base_name}_mask{i+1}.png")
    cv2.imwrite(save_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

# Append to CSV results
results.append({
    "image": os.path.basename(img_path),
    "bbox": f"[{x},{y},{w},{h}]",
    "mask1_score": float(scores[0]),
    "mask2_score": float(scores[1]),
    "mask3_score": float(scores[2]),
    "best_mask": int(best_idx + 1),
    "best_score": float(best_score)
})

# ---- Visualization ----
fig, axes = plt.subplots(1, 4, figsize=(20, 6))

# Original
axes[0].imshow(image_rgb)
axes[0].set_title("Original")
axes[0].axis("off")

# Preprocessed
axes[1].imshow(preprocessed_rgb)
axes[1].set_title("Preprocessed (CLAHE + Edges)")
axes[1].axis("off")

# Best Mask Overlay
axes[2].imshow(preprocessed_rgb)
axes[2].imshow(cleaned_masks[best_idx], alpha=0.5, cmap="jet")
axes[2].set_title(f"Best Mask (Score: {best_score:.2f})")
axes[2].axis("off")

# All masks
axes[3].imshow(preprocessed_rgb)
for i, mask in enumerate(cleaned_masks):
    axes[3].imshow(mask, alpha=0.3, cmap="jet")
axes[3].set_title("All Masks Overlay")
axes[3].axis("off")

plt.tight_layout()
plt.show()