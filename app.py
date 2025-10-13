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
import gc # Import garbage collector for manual cleanup

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="Cliniscan: Pneumonia AI Classifier",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL VARS ---
device = torch.device("cpu") # Explicitly use CPU 
class_names = ["NORMAL", "PNEUMONIA"]
MODEL_PATH = "model.pth" # PyTorch model path (Required for Grad-CAM)
ONNX_PATH = "model.onnx" # ONNX model path (Primary active classification model)

# DIRECT DOWNLOAD URL from the user's Google Drive link
# File ID: 1Ojg5BWXIiE0Z17kIJD33qEFAdTucgiMl
# Ensure this file is set to "Public" or "Anyone with the link" access.
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1Ojg5BWXIiE0Z17kIJD33qEFAdTucgiMl"

# --- MODEL ARCHITECTURE (Required for PyTorch Model Loading & Grad-CAM) ---
class PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaClassifier, self).__init__()
        # Use a pretrained ResNet50 as the base (assuming this was the base model)
        self.base_model = models.resnet50(weights=None)
        
        # Replace the final fully connected layer
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Define the feature extractor (all layers before the final FC layer)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        # Define the classifier (the final FC layer)
        self.classifier = self.base_model.fc

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- TRANSFORMS ---
# Standard transforms for inference
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- UTILITY FUNCTIONS ---

def get_status_color(class_name):
    """Returns 'red' for PNEUMONIA and 'green' for NORMAL for markdown text."""
    return "red" if class_name == "PNEUMONIA" else "green"

def create_conditional_bar_chart(df, model_name):
    """
    Creates a VERTICAL Altair bar chart with conditional colors for dark mode.
    Sets X-axis labels to black for visibility against light bars as requested.
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
        stroke='transparent'
    ).configure_title(
        fontSize=16,
        color='#ffffff'
    ).configure_axis(
        labelColor='#e0e0e0',
        titleColor='#e0e0e0',
        gridColor='#333333',
        domainColor='#e0e0e0'
    )
    
    return chart

# --- Grad-CAM UTILITIES (Requires PyTorch model) ---

def get_target_layer(model):
    """Dynamically finds the final convolutional layer for Grad-CAM (assuming ResNet50)."""
    if isinstance(model.base_model, models.ResNet):
        return model.base_model.layer4
    return model.features[-1] # Fallback for sequential models

def compute_grad_cam(model, input_tensor, target_layer):
    """Computes the Grad-CAM heatmap."""
    model.eval()
    
    activations = []
    gradients = []

    def save_activation(module, input, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks
    hook_handle_activation = target_layer.register_forward_hook(save_activation)
    hook_handle_gradient = target_layer.register_backward_hook(save_gradient)

    # Forward pass
    output = model(input_tensor)
    
    # Get the predicted class index
    pred_idx = output.argmax(dim=1).item()
    
    # Zero gradients and perform backward pass for the winning class
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0][pred_idx] = 1
    output.backward(gradient=one_hot, retain_graph=True)

    # Get the feature map and gradient
    A = activations[0].cpu().data.numpy()[0]
    W = gradients[0].cpu().data.numpy()[0]
    
    # Remove hooks
    hook_handle_activation.remove()
    hook_handle_gradient.remove()
    
    # Global average pooling of gradients
    weights = np.mean(W, axis=(1, 2))
    
    # Weighted combination
    cam = np.zeros(A.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * A[i, :, :]

    # ReLU and normalize
    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() > 0 else cam
    
    return cam

def overlay_heatmap(image_pil, heatmap):
    """Overlays the heatmap onto the original PIL image."""
    image_np = np.array(image_pil.convert("RGB"))

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    
    # Convert heatmap to 3-channel BGR image
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Convert original image to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Overlay: 60% original image, 40% heatmap
    superimposed_img = cv2.addWeighted(image_bgr, 0.6, heatmap_colored, 0.4, 0)
    
    # Convert back to RGB PIL Image
    return Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))


# --- MODEL LOADING (CACHED) ---

@st.cache_resource
def load_pytorch_model():
    """Attempts to load the PyTorch model for Grad-CAM use."""
    
    model = PneumoniaClassifier(num_classes=len(class_names)).to(device)
    
    if not os.path.exists(MODEL_PATH):
        st.warning(f"PyTorch model '{MODEL_PATH}' not found. Grad-CAM will be unavailable. Classification still uses ONNX.")
        return None
    
    try:
        # NOTE: This step is memory intensive and can cause crashes.
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        st.info("PyTorch model loaded successfully for Grad-CAM feature (Memory risk acknowledged).")
        return model
    except Exception as e:
        st.error(f"PyTorch model failed to load weights for Grad-CAM. Error: {e}")
        return None

@st.cache_resource
def load_onnx_model():
    """
    Loads the ONNX model session. If the file is not found, it attempts 
    to download it from the specified Google Drive link.
    """
    
    if not os.path.exists(ONNX_PATH):
        st.warning(f"Optimized ONNX model '{ONNX_PATH}' not found locally. Attempting to download from Google Drive...")
        
        try:
            with st.spinner(f"Downloading large model file ({ONNX_PATH}). This may take a moment..."):
                # Use curl command to download the file directly
                download_command = f"curl -L '{DRIVE_URL}' -o '{ONNX_PATH}'"
                exit_code = os.system(download_command)
                
                if exit_code != 0:
                    raise Exception(f"Download failed with exit code {exit_code}.")
                
            st.success(f"Model successfully downloaded as '{ONNX_PATH}'.")

        except Exception as e:
            st.error(f"Critical Error: Failed to download ONNX model from Google Drive. Ensure the shared link is public and accessible. Error: {e}")
            st.error("Application cannot run without a valid ONNX classification model.")
            return None

    # Proceed with loading if the file now exists (either found locally or downloaded)
    if os.path.exists(ONNX_PATH):
        file_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
        if file_size_mb < 0.1: # Minimal check for a corrupt/empty file (< 100 KB)
            st.error(f"Critical Error: Optimized ONNX model '{ONNX_PATH}' found but appears corrupt (Size: {file_size_mb:.2f} MB).")
            return None
            
        st.info(f"Model file '{ONNX_PATH}' found. Size: {file_size_mb:.2f} MB. Attempting to load.")

        try:
            # Rely on the pre-exported ONNX model for high-performance, low-memory inference
            return ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        except Exception as e:
            st.error(f"Failed to initialize ONNX Runtime session: {e}. Check if the file is a valid ONNX model.")
            return None
    
    return None


# --- LOAD MODELS ---
onnx_session = load_onnx_model()
pytorch_model = load_pytorch_model() # Load PyTorch model for Grad-CAM

# --- CHECK FOR CRITICAL ERRORS ---
if onnx_session is None:
    st.error("Application cannot run without a valid ONNX classification model.")
    st.stop()


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
    .st-emotion-cache-1ldfxyk, .st-emotion-cache-1y4v82y { 
        background-color: #000000; /* Pure Black */
        color: #ffffff;
    }
    /* Target the divider line in the sidebar for accent color */
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
    st.markdown("<p style='text-align: center; color: #cccccc;'>Pneumonia Classification powered by ONNX for stability</p>", unsafe_allow_html=True)
st.markdown("---")


# --- SIDEBAR FOR UPLOADER ---
with st.sidebar:
    st.header("Upload X-Ray Image")
    st.markdown("Upload a standard PA/AP chest X-ray image for AI-driven analysis.")
    uploaded_file = st.file_uploader("Choose a file (JPG, PNG)", type=["jpg","jpeg","png"])
    
    st.markdown("---")
    st.subheader("System Status")
    
    if pytorch_model:
        st.success("PyTorch Model Ready for Grad-CAM.")
    else:
        st.warning("PyTorch Model Unloaded (Grad-CAM Unavailable).")

    if onnx_session:
        st.success("ONNX Runtime Engine Ready for high-speed inference.")
    
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
    
    # Prepare tensor for ONNX inference
    img_tensor_onnx = transform(img).unsqueeze(0).numpy().astype(np.float32)
    
    # --- ONNX Prediction (The ONLY active classification model) ---
    outputs_onnx = onnx_session.run(None, {"input": img_tensor_onnx})[0][0]
    # Softmax for ONNX output
    probs_onnx = np.exp(outputs_onnx) / np.sum(np.exp(outputs_onnx))


    # --- Final Diagnosis (Using ONNX) ---
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

    # 3. Grad-CAM Image Replacement (Full Width - THIRD)
    st.subheader("Explainability Map (Grad-CAM)")
    
    if pytorch_model:
        st.info("Generating Grad-CAM heatmap using PyTorch model...")
        try:
            # Prepare tensor for PyTorch 
            img_tensor_pt = transform(img).unsqueeze(0).to(device)
            
            # Compute heatmap
            target_layer = get_target_layer(pytorch_model)
            heatmap = compute_grad_cam(pytorch_model, img_tensor_pt, target_layer)
            
            # Overlay heatmap
            grad_cam_img = overlay_heatmap(img, heatmap)
            
            st.image(grad_cam_img, caption="Activation Map (Grad-CAM)", use_container_width=True)
            st.success("Grad-CAM successfully generated, showing areas of highest model attention.")

        except Exception as e:
            st.error(f"Failed to generate Grad-CAM heatmap. Error: {e}")
            st.image("https://placehold.co/1000x300/1e1e1e/cf6679?text=Grad-CAM+Error+Occurred", use_container_width=True)

    else:
        # Warning if PyTorch model failed to load (either not found or memory issue)
        st.warning("""
        **Grad-CAM Feature Unavailable:** The full PyTorch model (`model.pth`) required for this explainability feature could not be loaded. This is often due to the large memory footprint of the PyTorch model. Classification is still running stably using the ONNX model.
        """)
        st.image("https://placehold.co/1000x300/1e1e1e/cf6679?text=Grad-CAM+Feature+Unavailable", use_container_width=True)

    st.markdown("---")
    st.caption("Disclaimer: AI models are assistive tools. Clinical interpretation is required.")
    
    gc.collect()


    # --- CONDITIONAL PREDICTION SCORES ---
    if show_scores:
        st.markdown("---")
        st.markdown("## 📊 Detailed Prediction Scores")

        # Only show one chart now, since only the ONNX model is active
        col_onnx, col_empty = st.columns(2)

        # ONNX Chart
        with col_onnx:
            st.markdown("### Optimized ONNX Prediction")
            color_onnx = get_status_color(final_diagnosis_class)

            st.markdown(
                f"Highest Confidence: **:{color_onnx}[{final_diagnosis_class}]** ({max_prob*100:.4f}%)"
            )
            df_onnx = pd.DataFrame({"Class": class_names, "Probability": probs_onnx.astype(float)})
        
            st.altair_chart(create_conditional_bar_chart(df_onnx, "ONNX Runtime"), use_container_width=True)

# --- NO FILE UPLOADED STATE (Initial Dashboard) ---
else:
    st.info("Upload an X-Ray image in the sidebar to begin the classification process.")

    st.markdown("## Application Overview")
    
    # Dashboard style placeholder
    st.markdown("""
        <div style="background-color: #1e1e1e; padding: 30px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5); text-align: center;">
            <h3 style="color: #03dac6; font-weight: 600;">Welcome to Cliniscan</h3>
            <p style="color: #cccccc;">An AI-powered tool for rapid preliminary classification of Pneumonia from chest X-rays.</p>
            <div style="margin-top: 20px; display: flex; justify-content: center; gap: 40px; color: #bb86fc;">
                <div>
                    <h1 style="font-size: 2em; margin: 0;'>🧠</h1>
                    <p style="margin: 5px 0 0 0; color: #cccccc;">AI Powered Diagnosis</p>
                </div>
                <div>
                    <h1 style="font-size: 2em; margin: 0;'>⚡</h1>
                    <p style="margin: 5px 0 0 0; color: #cccccc;">Optimized with ONNX</p>
                </div>
                <div>
                    <h1 style="font-size: 2em; margin: 0;'>🔒</h1>
                    <p style="margin: 5px 0 0 0; color: #cccccc;'>Stable & Memory-Efficient</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.image("https://placehold.co/1000x500/121212/bb86fc?text=Upload+an+Image+to+Start+Analysis", use_container_width=True, caption="Sample Chest X-Ray Placeholder")
    




