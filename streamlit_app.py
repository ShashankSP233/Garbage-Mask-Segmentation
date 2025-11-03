# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import torch
from segment_anything import sam_model_registry, SamPredictor
from streamlit_cropper import st_cropper
from PIL import Image

# ---------------- SETTINGS ----------------
st.set_page_config(page_title="SAM Segmentation UI", layout="wide")
checkpoint_path = "sam_vit_b_01ec64.pth"   # update with your model path
model_type = "vit_b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_sam():
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

predictor = load_sam()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")
image_folder = st.sidebar.text_input("Image Folder", "images")
export_csv = st.sidebar.button("Export Results to CSV")

# Storage for results
if "results" not in st.session_state:
    st.session_state.results = []

# ---------------- MAIN ----------------
st.title("SAM Segmentation with ROI Selection")

if not os.path.exists(image_folder):
    st.warning(f"⚠️ Folder '{image_folder}' not found.")
else:
    files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    if not files:
        st.warning("No images found in folder.")
    else:
        img_index = st.sidebar.number_input("Image Index", 0, len(files)-1, 0)
        image_path = os.path.join(image_folder, files[img_index])
        st.subheader(f"Image: {files[img_index]}")

        # Load image
        pil_img = Image.open(image_path).convert("RGB")
        
        # ROI Selection
        st.write("### Draw ROI")
        cropped_img = st_cropper(pil_img, realtime_update=True, aspect_ratio=None, box_color='red')
        
        # Convert to numpy
        image_rgb = np.array(pil_img)
        predictor.set_image(image_rgb)

        if st.button("Run SAM on ROI"):
            # Map crop back to original image
            # Get bbox coords
            box = st.session_state["cropper_box"] if "cropper_box" in st.session_state else None
            if box is None:
                st.error("Please select a region first.")
            else:
                x, y, w, h = box
                input_box = np.array([x, y, x+w, y+h])
                
                masks, scores, _ = predictor.predict(
                    box=input_box[None, :],
                    multimask_output=True
                )
                
                # Display masks
                for i in range(len(masks)):
                    st.write(f"**Mask {i+1} (Score: {scores[i]:.3f})**")
                    overlay = image_rgb.copy()
                    mask = masks[i].astype(np.uint8) * 255
                    overlay[mask > 0] = [255, 0, 0]  # red overlay
                    
                    st.image(overlay, caption=f"Mask {i+1}", use_column_width=True)
                
                # Save results
                st.session_state.results.append({
                    "image": files[img_index],
                    "mask1_score": float(scores[0]),
                    "mask2_score": float(scores[1]),
                    "mask3_score": float(scores[2])
                })

if export_csv:
    if st.session_state.results:
        df = pd.DataFrame(st.session_state.results)
        csv_path = "sam_results.csv"
        df.to_csv(csv_path, index=False)
        st.success(f"✅ Results saved to {csv_path}")
        st.dataframe(df)
    else:
        st.warning("No results yet.")
