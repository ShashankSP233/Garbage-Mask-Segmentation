import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import torch

# ---- Load image ----
image_path = "sample.jpg"
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Resize for display
display_height = 720
scale = display_height / image_rgb.shape[0]
resized_image = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

# ---- USER SELECTS REGION ----
bbox = cv2.selectROI("Select Region (Press ENTER)", resized_image, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Convert scaled box back to original size
x_scaled, y_scaled, w_scaled, h_scaled = bbox
x = int(x_scaled / scale)
y = int(y_scaled / scale)
w = int(w_scaled / scale)
h = int(h_scaled / scale)
input_box = np.array([x, y, x + w, y + h])

# ---- Load SAM Model ----
checkpoint_path = "sam_vit_b_01ec64.pth"  # update with your path
model_type = "vit_b"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# ---- Predict masks ----
masks, scores, _ = predictor.predict(
    box=input_box[None, :],
    multimask_output=True
)

# ---- Clean masks ----
def clean_mask(mask, img_shape):
    h, w = img_shape[:2]
    k = max(3, min(h, w) // 200)  # adaptive kernel size
    kernel = np.ones((k, k), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned

cleaned_masks = [clean_mask(m, image_rgb.shape) for m in masks]

# ---- Pick best mask ----
best_idx = np.argmax(scores)
best_mask = cleaned_masks[best_idx]
print(f"Best mask chosen with score: {scores[best_idx]:.4f}")

# ---- Visualization ----
# Draw bounding box
vis_img = image_rgb.copy()
cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(8, 6))
plt.imshow(vis_img)
plt.imshow(best_mask, alpha=0.5, cmap='jet')
plt.title(f"Best Mask (Score: {scores[best_idx]:.2f})")
plt.axis('off')
plt.show()

# ---- Save masks ----
for i, mask in enumerate(cleaned_masks):
    rgba = np.dstack((image_rgb, (mask * 255).astype(np.uint8)))
    cv2.imwrite(f"mask_{i+1}.png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
print("Masks saved as mask_1.png, mask_2.png, ...")
