import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import onnxruntime as ort
import numpy as np
import altair as alt
import os
import onnx
import cv2
import io
from pathlib import Path

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="Cliniscan: Pneumonia AI Classifier",
    page_icon="⚕️",
    layout="wide", # Use wide layout for a better dashboard feel
    initial_sidebar_state="expanded"
)

# --- GLOBAL VARS ---
device = torch.device("cpu")
class_names = ["NORMAL", "PNEUMONIA"]
MODEL_PATH = "model.pth" # Using the correct file name found in your repository

# --- TRANSFORMS (Kept the same for functionality) ---
transform_gradcam = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- UTILITY FUNCTIONS ---

def generate_grad_cam(model, target_layer, img_tensor, original_img):
    """Generates a heat map using Grad-CAM."""
    model.eval()
    target_class_idx = 1
    
    feature_maps = []
    def forward_hook(module, input, output):
        feature_maps.append(output)
    
    gradients = []
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    hook_handle_fm = target_layer.register_forward_hook(forward_hook)
    hook_handle_grad = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    model.zero_grad()
    target_score = output[0, target_class_idx]
    # Perform backward pass only if we have a target score
    if target_score.numel() > 0:
        target_score.backward(retain_graph=True)
    
    hook_handle_fm.remove()
    hook_handle_grad.remove()
    
    if not gradients:
        return Image.new('RGB', (original_img.width, original_img.height), color = 'gray')

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    feature_map = feature_maps[0].squeeze(0)
    for i in range(feature_map.shape[0]):
        feature_map[i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(feature_map, dim=0).relu()
    
    heatmap /= torch.max(heatmap)
    heatmap_np = heatmap.detach().cpu().numpy()
    
    heatmap_resized = cv2.resize(heatmap_np, (original_img.width, original_img.height))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    img_np = np.array(original_img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    superimposed_img = cv2.addWeighted(img_bgr, 0.5, heatmap_colored, 0.5, 0)
    
    superimposed_img_pil = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    
    return superimposed_img_pil


def get_status_color(class_name):
    """Returns 'red' for PNEUMONIA and 'green' for NORMAL for markdown text."""
    # Using Streamlit's built-in color strings
    return "red" if class_name == "PNEUMONIA" else "green"

def create_conditional_bar_chart(df, model_name):
    """
    Creates a VERTICAL Altair bar chart with conditional colors for dark mode.
    Sets X-axis labels to black for visibility against light bars.
    """
    
    color_scale = alt.condition(
        alt.datum.Class == "PNEUMONIA",
        alt.value("#cf6679"), # Dark mode soft red for warning
        alt.value("#03dac6")  # Dark mode teal for success
    )
    
    chart = alt.Chart(df).mark_bar(size=40).encode(
        # Set X-axis label color explicitly to black for visibility
        x=alt.X("Class", sort="-y", title=None, axis=alt.Axis(labels=True, title=None, labelColor="#000000")),
        y=alt.Y("Probability", axis=alt.Axis(format=".0%", title="Probability")),
        color=color_scale,
        tooltip=["Class", alt.Tooltip("Probability", format=".4%")]
    ).properties(
        title=f"{model_name} Class Probabilities"
    ).interactive()
    
    # Apply dark mode configurations
    chart = chart.configure_view(
        stroke='transparent' # Remove chart border
    ).configure_title(
        fontSize=16,
        color='#ffffff' # White title color
    ).configure_axis(
        labelColor='#e0e0e0', # Light grey axis labels (Y-axis will use this)
        titleColor='#e0e0e0', # Light grey axis titles
        gridColor='#333333',  # Dark grid lines
        domainColor='#e0e0e0' # Axis line color
    )
    
    return chart

# --- MODEL LOADING (CASHED) ---

@st.cache_resource
def load_pytorch_model():
    """Loads the trained PyTorch model structure and weights."""
    
    if not Path(MODEL_PATH).exists():
        st.error(f"Model file not found: '{MODEL_PATH}'. Please ensure it is uploaded and tracked with Git LFS.")
        return None

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    # Using the correct model name "model.pth"
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model weights from '{MODEL_PATH}'. Check file integrity and Git LFS status: {e}")
        return None


pytorch_model = load_pytorch_model()

@st.cache_resource
def load_onnx_model(_model_pt, device):
    """Loads or exports the ONNX model."""
    ONNX_PATH = "model.onnx"
    
    if not os.path.exists(ONNX_PATH) or os.path.getsize(ONNX_PATH) < 100000:
        
        st.warning(f"ONNX model '{ONNX_PATH}' not found or is a placeholder. Attempting to export from PyTorch...")
        
        dummy_input = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)

        try:
            torch.onnx.export(
                _model_pt, 
                dummy_input,
                ONNX_PATH,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                              'output': {0: 'batch_size'}}
            )
            onnx_model = onnx.load(ONNX_PATH)
            onnx.checker.check_model(onnx_model)
            st.success("Successfully exported and checked ONNX model.")
        except Exception as e:
            st.error(f"Failed to export ONNX model: {e}")
            return None 

    return ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# --- RUN MODEL LOADING AND CHECK FOR ERRORS ---
if pytorch_model is None:
    st.stop()

onnx_session = load_onnx_model(pytorch_model, device)

# --- STREAMLIT UI/LAYOUT ---

# Custom CSS for dark mode aesthetics, sidebar, and accent divider
st.markdown("""
<style>
    /* Set the main app background to a dark color and text to white */
    .stApp {
        background-color: #121212; /* Deep Charcoal Background */
        color: white;
    }
    
    /* Main title styling */
    .stApp > header {
        background-color: transparent;
    }
    h1 {
        font-size: 2.5em;
        color: #bb86fc; /* Light Purple/Magenta accent for dark mode */
        text-align: center;
        margin-bottom: 0.5em;
        font-weight: 700;
    }
    /* Default text color for the title markdown description */
    p {
        color: #e0e0e0;
    }
    .stAlert {
        border-radius: 10px;
    }
    /* Section headers */
    .stMarkdown h2 {
        color: #03dac6; /* Teal accent for dark mode */
        font-weight: 500;
        border-bottom: 2px solid #333333; /* Darker border */
        padding-bottom: 5px;
        margin-top: 30px;
    }
    
    /* Diagnostic Result Box Styling */
    .main-result-box {
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4); /* Stronger shadow for dark mode */
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* PNEUMONIA Warning Box */
    .pneumonia-detected {
        background-color: #2b1c1e; /* Dark background that matches the soft red accent */
        border-left: 8px solid #cf6679; /* Soft red border */
        color: white;
    }
    .pneumonia-detected h3, .pneumonia-detected p {
        color: white;
    }
    
    /* NORMAL Finding Box */
    .normal-finding {
        background-color: #1b2d2d; /* Dark background that matches the teal accent */
        border-left: 8px solid #03dac6; /* Teal border */
        color: white;
    }
    .normal-finding h3, .normal-finding p {
        color: white;
    }
    
    /* Sidebar styling: Pure Black background and accent divider */
    /* Targeting the main sidebar element and its content */
    /* Note: These selectors are based on current Streamlit class names and might need adjustment if Streamlit changes. */
    .st-emotion-cache-1ldfxyk, .st-emotion-cache-1y4v82y { 
        background-color: #000000; /* Pure Black */
        color: #ffffff;
    }
    /* Targeting the divider line in the sidebar */
    .st-emotion-cache-1ldfxyk > div:first-child > div:nth-child(2) {
        border-top: 2px solid #bb86fc; /* Light Purple/Magenta accent for divider */
    }
    /* Ensuring sidebar text remains white */
    .st-emotion-cache-1d3w5hv {
        color: #ffffff;
    }
    
    /* Hide the default Streamlit footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- MAIN HEADER ---
col_logo, col_title, col_info = st.columns([1, 4, 1])

with col_title:
    st.title("⚕️ Cliniscan: Chest X-Ray AI Assistant")
    st.markdown("<p style='text-align: center; color: #cccccc;'>Pneumonia Classification & Explainability using PyTorch & ONNX</p>", unsafe_allow_html=True)
st.markdown("---")


# --- SIDEBAR FOR UPLOADER ---
with st.sidebar:
    st.header("Upload X-Ray Image")
    st.markdown("Upload a standard PA/AP chest X-ray image for AI-driven analysis.")
    uploaded_file = st.file_uploader("Choose a file (JPG, PNG)", type=["jpg","jpeg","png"])
    
    st.markdown("---")
    st.subheader("System Status")
    
    # Show loading status for models
    if pytorch_model:
        st.success("PyTorch Model (ResNet-18) Loaded.")
    else:
        st.error("PyTorch Model Failed to Load.")

    if onnx_session:
        st.success("ONNX Runtime Engine Ready.")
    else:
        st.warning("ONNX Loading Failed. Falling back to PyTorch-only inference.")
    
    st.markdown("---")
    
    # New element: Checkbox to control visibility of prediction scores
    show_scores = st.checkbox("Show Detailed Prediction Scores", value=False) 
    
    st.markdown("---")
    st.caption("Developed for educational & research purposes.")


# --- CORE APPLICATION LOGIC ---
if uploaded_file:
    
    # --- Data Prep ---
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Use the globally cached pytorch_model. target_layer must be defined inside the block.
    target_layer = pytorch_model.layer4[-1].conv2 
    
    img_tensor = transform(img).unsqueeze(0).to(device).to(torch.float32) 
    gradcam_tensor = transform_gradcam(img).unsqueeze(0).to(device).to(torch.float32)
    gradcam_tensor.requires_grad_(True)
    
    # --- PyTorch Prediction (Used for Grad-CAM logic) ---
    outputs_pt = pytorch_model(img_tensor)
    probs_pt = torch.softmax(outputs_pt, dim=1)[0]
    pred_class_pt = class_names[torch.argmax(probs_pt).item()]

    # --- ONNX Prediction (Used for Final Diagnosis) ---
    if onnx_session:
        x = transform(img).unsqueeze(0).numpy().astype(np.float32)
        outputs_onnx = onnx_session.run(None, {"input": x})[0][0]
        # Softmax for ONNX output
        probs_onnx = np.exp(outputs_onnx) / np.sum(np.exp(outputs_onnx))
    else:
        # Fallback to PyTorch prediction if ONNX failed
        probs_onnx = probs_pt.detach().cpu().numpy()


    # --- Final Diagnosis (Using ONNX/Fallback) ---
    final_pred_idx = np.argmax(probs_onnx)
    final_diagnosis_class = class_names[final_pred_idx]
    max_prob = probs_onnx[final_pred_idx]
    
    
    # --- DIAGNOSIS & VISUALIZATION (VERTICAL STACK) ---
    st.markdown("## 🔬 Diagnosis & Visualization")

    # 1. Classification Message (Full Width - FIRST)
    st.subheader("Classification Result")
    
    if final_diagnosis_class == "PNEUMONIA":
        box_class = "pneumonia-detected"
        icon = "🚨"
        message = "⚠️ **IMMEDIATE ACTION:** Consultation with a medical professional is strongly recommended."
        st.error(message)
    else:
        box_class = "normal-finding"
        icon = "✅"
        message = "**Model Finding:** No Sign of Pneumonia Detected on this image."
        st.success(message)
        
    st.markdown(f"""
    <div class="main-result-box {box_class}">
        <h3>{icon} {final_diagnosis_class}</h3>
        <p style="font-size: 1.5em; font-weight: bold;">
            Confidence: {max_prob*100:.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # 2. Uploaded Image (Full Width - SECOND)
    st.subheader("Uploaded X-ray")
    st.image(img, use_container_width=True)
    
    st.markdown("---")

    # 3. Grad-CAM Image (Full Width - THIRD)
    st.subheader("Model Focus (Grad-CAM)")
    if pred_class_pt == "PNEUMONIA":
        with st.spinner("Generating Explainability Heatmap..."):
            # Use PyTorch model for Grad-CAM as ONNX doesn't support backward hooks
            heatmap_img = generate_grad_cam(pytorch_model, target_layer, gradcam_tensor, img)
            st.image(heatmap_img, caption="Areas contributing to diagnosis (Red/Yellow)", use_container_width=True)
    else:
        st.image(img, caption="Grad-CAM visualization (Model decided finding is normal)", use_container_width=True)
        st.info("Grad-CAM is typically most useful for positive findings (e.g., PNEUMONIA).")
        
    st.markdown("---")
    st.caption("Disclaimer: AI models are assistive tools. Clinical interpretation is required.")


    # --- CONDITIONAL PREDICTION SCORES ---
    if show_scores:
        st.markdown("---")
        st.markdown("## 📊 Detailed Prediction Scores")

        col_pt, col_onnx = st.columns(2)

        # PyTorch Chart
        with col_pt:
            st.markdown("### PyTorch Prediction")
            pred_idx_pt = torch.argmax(probs_pt).item()
            pred_class_pt_display = class_names[pred_idx_pt]
            pred_prob_pt = probs_pt[pred_idx_pt].item()
            color_pt = get_status_color(pred_class_pt_display)

            st.markdown(
                f"Highest Confidence: **:{color_pt}[{pred_class_pt_display}]** ({pred_prob_pt*100:.4f}%)"
            )
            df_pt = pd.DataFrame({"Class": class_names, "Probability":[p.item() for p in probs_pt]})
        
            st.altair_chart(create_conditional_bar_chart(df_pt, "PyTorch"), use_container_width=True) 

        # ONNX Chart
        with col_onnx:
            st.markdown("### ONNX Prediction (Optimized)")
            pred_prob_onnx = probs_onnx[final_pred_idx]
            color_onnx = get_status_color(final_diagnosis_class)

            st.markdown(
                f"Highest Confidence: **:{color_onnx}[{final_diagnosis_class}]** ({pred_prob_onnx*100:.4f}%)"
            )
            df_onnx = pd.DataFrame({"Class": class_names, "Probability": probs_onnx.astype(float)})
        
            st.altair_chart(create_conditional_bar_chart(df_onnx, "ONNX Runtime"), use_container_width=True)

# --- NO FILE UPLOADED STATE (Initial Dashboard) ---
else:
    st.info("Upload an X-Ray image in the sidebar to begin the classification process.")

    st.markdown("## Application Overview")
    
    # Dashboard style placeholder (similar to Image 2)
    st.markdown("""
        <div style="background-color: #1e1e1e; padding: 30px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5); text-align: center;">
            <h3 style="color: #03dac6; font-weight: 600;">Welcome to Cliniscan</h3>
            <p style="color: #cccccc;">An AI-powered tool for rapid preliminary classification of Pneumonia from chest X-rays.</p>
            <div style="margin-top: 20px; display: flex; justify-content: center; gap: 40px; color: #bb86fc;">
                <div>
                    <h1 style="font-size: 2em; margin: 0;">🧠</h1>
                    <p style="margin: 5px 0 0 0; color: #cccccc;">AI Powered Diagnosis</p>
                </div>
                <div>
                    <h1 style="font-size: 2em; margin: 0;">⚡</h1>
                    <p style="margin: 5px 0 0 0; color: #cccccc;">Optimized with ONNX</p>
                </div>
                <div>
                    <h1 style="font-size: 2em; margin: 0;">🔍</h1>
                    <p style="margin: 5px 0 0 0; color: #cccccc;">Grad-CAM Explainability</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.image("https://placehold.co/1000x500/121212/bb86fc?text=Upload+an+Image+to+Start+Analysis", use_container_width=True, caption="Sample Chest X-Ray Placeholder")










