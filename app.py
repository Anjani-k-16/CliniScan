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
# Set device to CPU as models are currently stubbed/mocked
device = torch.device("cpu") 
class_names = ["NORMAL", "PNEUMONIA"]
MODEL_PATH = "model.pth" # Placeholder path

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
    """Generates a heat map using Grad-CAM. (STUBBED FOR NOW)"""
    st.warning("Note: Grad-CAM is using a PyTorch model stub and will not display correctly until the real model is loaded.")
    # Return the original image or a simple gray placeholder while stubbing the model logic
    return Image.fromarray(np.array(original_img.convert('RGB')))

def get_status_color(class_name):
    """Returns 'red' for PNEUMONIA and 'green' for NORMAL for markdown text."""
    return "red" if class_name == "PNEUMONIA" else "green"

def create_conditional_bar_chart(df, model_name):
    """
    Creates a VERTICAL Altair bar chart with conditional colors for dark mode.
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

# --- MOCK CLASSES AND FUNCTIONS FOR STUBBING ---

class MockModel:
    """A dummy class to replace the PyTorch model for UI testing."""
    def __init__(self):
        # Create a mock structure for Grad-CAM to avoid AttributeError
        class MockLayer:
            def __init__(self):
                self.conv2 = None
        self.layer4 = [MockLayer()]
        
    def eval(self):
        pass
    def to(self, device):
        return self
    def load_state_dict(self, state_dict):
        pass
    def __call__(self, x):
        # Return random logit-like output for simulation
        # This function is now mostly bypassed in the prediction logic below
        return torch.tensor([[0.5, -0.5]], dtype=torch.float32)

@st.cache_resource
def load_pytorch_model():
    """Loads the trained PyTorch model structure and weights. (STUBBED)"""
    # Since the file is missing, we return a mock model.
    st.error(f"Model file not found: '{MODEL_PATH}'. Using a **MOCK MODEL** for UI demonstration.")
    return MockModel()

@st.cache_resource
def load_onnx_model(_model_pt, device):
    """Loads or exports the ONNX model. (STUBBED)"""
    st.error(f"ONNX model loading skipped due to missing PyTorch model.")
    return None

# --- RUN MODEL LOADING AND CHECK FOR ERRORS ---
# We still call these, but they now return the mock objects
pytorch_model = load_pytorch_model()
onnx_session = load_onnx_model(pytorch_model, device)

# --- STREAMLIT UI/LAYOUT ---

# Custom CSS for dark mode aesthetics, sidebar, and white cast removal
st.markdown("""
<style>
    /* ------------------------------------------- */
    /* GLOBAL DARK MODE FIXES (White Cast Removal) */
    /* ------------------------------------------- */
    /* Set the main app background to a dark color and text to white */
    .stApp {
        background-color: #121212; /* Deep Charcoal Background */
        color: white;
    }
    /* Target the core Streamlit container elements to enforce the dark background */
    div[data-testid="stAppViewContainer"],
    div[data-testid="stVerticalBlock"] > div {
        background-color: #121212 !important;
    }
    /* Ensure markdown container uses dark background */
    div[data-testid="stMarkdownContainer"] {
        background-color: transparent;
    }
    body {
        background-color: #121212 !important;
    }

    /* ------------------------------------------- */
    /* CUSTOM STYLING */
    /* ------------------------------------------- */

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
    div[data-testid="stSidebarContent"] {
        background-color: #000000 !important; /* Enforce Pure Black */
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
    if isinstance(pytorch_model, MockModel):
        st.warning("PyTorch Model (Mock) Active.")
        
        # === NEW: Simulation Control for UI Testing ===
        st.markdown("---")
        st.subheader("Simulation Control")
        # Radio button to let the user select the forced outcome
        mock_result_choice = st.radio(
            "Force Simulated Diagnosis:",
            ("NORMAL", "PNEUMONIA"),
            index=0,
            key="mock_result"
        )
        # ===============================================
        
    else:
        st.success("PyTorch Model (ResNet-18) Loaded.")

    if onnx_session is None:
        st.warning("ONNX Runtime Engine Inactive.")
    else:
        st.success("ONNX Runtime Engine Ready.")
    
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
    # Note: We can only call this if we have the real model, but mock model has the structure
    target_layer = pytorch_model.layer4[-1].conv2 
    
    img_tensor = transform(img).unsqueeze(0).to(device).to(torch.float32) 
    gradcam_tensor = transform_gradcam(img).unsqueeze(0).to(device).to(torch.float32)
    gradcam_tensor.requires_grad_(True)
    
    # --- Prediction Simulation ---
    if isinstance(pytorch_model, MockModel):
        # Read the user-controlled simulation setting
        mock_result_choice = st.session_state.mock_result
        
        # Generate FIXED, high-confidence probabilities based on the user's choice
        if mock_result_choice == "PNEUMONIA":
            # Simulate PNEUMONIA result (e.g., 98% confidence)
            probs_onnx = np.array([0.02, 0.98], dtype=np.float32)
        else:
            # Simulate NORMAL result (e.g., 98% confidence)
            probs_onnx = np.array([0.98, 0.02], dtype=np.float32)
        
        # PyTorch result is the same as ONNX for stubbing simplicity
        probs_pt = torch.tensor(probs_onnx)
    else:
        # --- PyTorch Prediction (Used for Grad-CAM logic) ---
        outputs_pt = pytorch_model(img_tensor)
        probs_pt = torch.softmax(outputs_pt, dim=1)[0]

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
    
    # We still need a pred_class_pt for the Grad-CAM conditional
    pred_class_pt = class_names[torch.argmax(probs_pt).item()]
    
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
    if pred_class_pt == "PNEUMONIA" and not isinstance(pytorch_model, MockModel):
        with st.spinner("Generating Explainability Heatmap..."):
            # Use PyTorch model for Grad-CAM as ONNX doesn't support backward hooks
            heatmap_img = generate_grad_cam(pytorch_model, target_layer, gradcam_tensor, img)
            st.image(heatmap_img, caption="Areas contributing to diagnosis (Red/Yellow)", use_container_width=True)
    elif isinstance(pytorch_model, MockModel):
        st.image(img, caption="Grad-CAM visualization is disabled (Mock Model active)", use_container_width=True)
        st.warning("The real model must be loaded to generate the Grad-CAM visualization. **Simulated Diagnosis:** " + st.session_state.mock_result)
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
    
    col_img, col_text = st.columns([2, 3])

    with col_img:
        # Placeholder for Lung Image (using a neutral X-ray image URL)
        # Using a professional placeholder to suggest a chest X-ray
        st.image("https://placehold.co/400x400/1e1e1e/bb86fc?text=CHEST+X-RAY", 
                 caption="AI Classification in Action", use_container_width=True)

    with col_text:
        # Cleaned up the text structure and removed emojis for a professional look
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #1e1e1e;">
                <h3 style="color: #bb86fc; font-weight: 600; margin-top: 0;">Cliniscan: Pneumonia AI Classifier</h3>
                <p style="color: #cccccc;">This tool utilizes a deep learning model (ResNet-18) trained on thousands of chest X-ray images to provide rapid, preliminary classification for Pneumonia and Normal findings.</p>
                
                <h4 style="color: #03dac6; margin-top: 20px;">Key Features:</h4>
                <ul style="color: #e0e0e0; padding-left: 20px;">
                    <li>**Fast Inference:** Optimized with the ONNX Runtime for low-latency predictions.</li>
                    <li>**Explainability:** Uses Grad-CAM to visualize the exact regions (heatmaps) of the X-ray the model focused on for its diagnosis.</li>
                    <li>**High Confidence:** Provides classification confidence scores for both 'NORMAL' and 'PNEUMONIA' classes.</li>
                </ul>
                <p style="color: #cf6679; font-weight: bold; margin-top: 20px;">Note: This is an assistive tool for research and education. Always rely on clinical judgment.</p>
            </div>
            """, unsafe_allow_html=True
        )



