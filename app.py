import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import altair as alt 
import os 
import cv2 
import io 
# Removed gdown as we are not downloading large weights anymore

# --- Configuration ---
device = torch.device("cpu") 

# --- Model Artifact Location (Placeholder for stability) ---
MODEL_PTH_PATH = "model_stable.pth"
GOOGLE_DRIVE_FILE_ID = "1FN8UG5pJiKPT8_yE8CkvlTC2DthCxbVR"
# -----------------------------


transform_gradcam = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class_names = ["NORMAL", "PNEUMONIA"]


def generate_grad_cam(model, target_layer, img_tensor, original_img):
    """
    MOCK Grad-CAM: Since the model is a mock and has no meaningful weights, 
    this function will return a simple gray image to prevent errors 
    and show the UI flow.
    """
    return Image.new('RGB', (original_img.width, original_img.height), color = 'gray')


def get_status_color(class_name):
    """Returns 'red' for PNEUMONIA and 'green' for NORMAL for markdown text."""
    return "red" if class_name == "PNEUMONIA" else "green"

def create_conditional_bar_chart(df, model_name):
    """Creates a VERTICAL Altair bar chart with conditional colors."""
    
    color_scale = alt.condition(
        alt.datum.Class == "PNEUMONIA",
        alt.value("red"),
        alt.value("green") 
    )

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Class", sort="-y", title=None, axis=alt.Axis(labels=True, title=None)), 
        y=alt.Y("Probability", axis=alt.Axis(format=".0%", title="Probability")),
        color=color_scale,
        tooltip=["Class", alt.Tooltip("Probability", format=".4%")] 
    ).properties(
        title=f"{model_name} Class Probabilities" 
    ).interactive()
    
    return chart


@st.cache_resource
def load_pytorch_model():
    """
    *** MOCK MODEL LOAD ***
    Loads a standard ResNet18 structure without requiring external weight files 
    or resource-heavy downloads, ensuring the application successfully launches.
    """
    
    with st.spinner("Loading MOCK PyTorch Model Structure..."):
        # Load ResNet18 structure with NO PRE-TRAINED WEIGHTS (weights=None)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        st.success("Mock Model Loaded. Classification results will be random.")
        
    model.to(device)
    model.eval()
    return model

pytorch_model = load_pytorch_model()

# The model is now guaranteed to load, so no need for st.stop() here.


# --- Streamlit UI ---

st.markdown(
    """
    <style>
    .stApp { background-color: #000000; color: white; }
    [data-testid="stVerticalBlock"] { background-color: #000000; }
    h1, h2, h3, h4, p, label { color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("CHEST X-RAY CLASSIFIER (MOCK DEMO MODE)") 
st.warning("Resource limits prevented loading the full model. This is a functional UI test with random predictions.")
st.markdown("---")


uploaded_file = st.file_uploader("Choose a Chest X-ray image", type=["jpg","jpeg","png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # We use the cached model, but define target layer here
    current_model = pytorch_model 
    target_layer = current_model.layer4[-1].conv2 
    
    img_tensor = transform(img).unsqueeze(0).to(device).to(torch.float32) 
    gradcam_tensor = transform_gradcam(img).unsqueeze(0).to(device).to(torch.float32)
    gradcam_tensor.requires_grad_(True)
    
    
    # --- MOCK PREDICTION ---
    # Since the model has random weights, we just run a forward pass
    with torch.no_grad():
        outputs_pt = current_model(img_tensor)
    
    probs_pt = torch.softmax(outputs_pt, dim=1)[0]
    pred_idx_pt = torch.argmax(probs_pt).item()
    pred_class_pt = class_names[pred_idx_pt]

    # --- FINAL DIAGNOSIS ---
    final_diagnosis_class = pred_class_pt
    max_prob = probs_pt[pred_idx_pt].item()

    final_diagnosis_color = get_status_color(final_diagnosis_class)
    
    
    st.markdown("### Classification Result:")
    
 
    if final_diagnosis_class == "PNEUMONIA":
        st.markdown(f"## :{final_diagnosis_color}[MOCK PNEUMONIA DETECTED]")
        st.warning("Action Recommended: This is a random result. Do not use for medical advice.")
    else:
        st.markdown(f"## :{final_diagnosis_color}[MOCK NORMAL FINDING]")
    
        st.info("**Model Analysis:** Results are random as external weights could not be loaded.") 

    st.markdown(f"Confidence: **{max_prob*100:.2f}%**")
    st.markdown("---")

    st.markdown("### Uploaded X-ray")
    st.image(img, use_container_width=True)
    st.markdown("---")

    
    st.markdown("### Model Focus (Grad-CAM Visualization)")
    with st.spinner("Generating MOCK Explainability Heatmap..."):
        # The mock function returns a gray box
        heatmap_img = generate_grad_cam(current_model, target_layer, gradcam_tensor, img)
        st.image(heatmap_img, caption="MOCK: Visualization is a simple gray image.", use_container_width=True)

    st.markdown("---")

    
 
    st.markdown("### Model Prediction Scores (MOCK)")
    
    df_pt = pd.DataFrame({"Class": class_names, "Probability":[p.item() for p in probs_pt]})
    st.altair_chart(create_conditional_bar_chart(df_pt, "MOCK PyTorch"), use_container_width=True)
    





















