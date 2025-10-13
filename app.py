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
import warnings

# Suppress harmless PyTorch UserWarnings that clutter the console
warnings.filterwarnings("ignore", category=UserWarning)

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
MODEL_PATH = "model.pth" # PyTorch model path
ONNX_PATH = "model.onnx" # ONNX model path

# --- IMPORTANT: MODEL DOWNLOAD URLs ---
# YOU MUST REPLACE THIS ID WITH THE PUBLIC GOOGLE DRIVE ID OF YOUR PYTORCH MODEL.
PT_MODEL_DRIVE_URL = "https://drive.google.com/uc?export=download&id=YOUR_PYTORCH_MODEL_ID" 
# NOTE: The ONNX model is exported below, so it doesn't need an external URL unless it's too big to export.

# --- TRANSFORMS (Kept the same for functionality) ---
# Normalization transform for prediction (ONNX/PyTorch)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Non-normalized transform for Grad-CAM input tensor
transform_gradcam = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- UTILITY FUNCTIONS ---

def generate_grad_cam(model, target_layer, img_tensor, original_img):
    """Generates a heat map using Grad-CAM."""
    model.eval()
    # Assuming we are calculating Grad-CAM for the PNEUMONIA class (index 1)
    target_class_idx = 1
    
    feature_maps = []
    def forward_hook(module, input, output):
        # Ensure we detach before storing
        feature_maps.append(output.detach()) 
    
    gradients = []
    def backward_hook(module, grad_in, grad_out):
        # Ensure we detach before storing
        gradients.append(grad_out[0].detach())

    # Ensure tensor requires gradient for backward pass
    img_tensor.requires_grad_(True) 

    hook_handle_fm = target_layer.register_forward_hook(forward_hook)
    hook_handle_grad = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    model.zero_grad()
    
    # Check if the model predicted PNEUMONIA for target score.
    # We only care about the gradient of the predicted class or the target class (PNEUMONIA=1)
    target_score = output[0, target_class_idx]
    
    if target_score.numel() > 0:
        target_score.backward(retain_graph=True)
    
    hook_handle_fm.remove()
    hook_handle_grad.remove()
    
    # Safety check for missing gradients (e.g., if forward pass failed)
    if not gradients or not feature_maps:
        return Image.new('RGB', (original_img.width, original_img.height), color = 'gray')

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    feature_map = feature_maps[0].squeeze(0)
    
    # Weighted combination of feature maps
    for i in range(feature_map.shape[0]):
        feature_map[i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(feature_map, dim=0).relu()
    
    # Normalize the heatmap
    max_val = torch.max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    else:
        heatmap = torch.zeros_like(heatmap) # Avoid division by zero
        
    heatmap_np = heatmap.cpu().numpy()
    
    # Resize and overlay
    heatmap_resized = cv2.resize(heatmap_np, (original_img.width, original_img.height))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    img_np = np.array(original_img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Blend the original image and the heatmap
    superimposed_img = cv2.addWeighted(img_bgr, 0.5, heatmap_colored, 0.5, 0)
    
    superimposed_img_pil = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    
    return superimposed_img_pil


def get_status_color(class_name):
    """Returns 'red' for PNEUMONIA and 'green' for NORMAL for markdown text."""
    return "red" if class_name == "PNEUMONIA" else "green"

def create_conditional_bar_chart(df, model_name):
    """Creates a VERTICAL Altair bar chart with conditional colors for dark mode."""
    
    color_scale = alt.condition(
        alt.datum.Class == "PNEUMONIA",
        alt.value("#cf6679"), # Dark mode soft red for warning
        alt.value("#03dac6")  # Dark mode teal for success
    )
    
    chart = alt.Chart(df).mark_bar(size=40).encode(
        x=alt.X("Class", sort="-y", title=None, axis=alt.Axis(labels=True, title=None)),
        y=alt.Y("Probability", axis=alt.Axis(format=".0%", title="Probability")),
        color=color_scale,
        tooltip=["Class", alt.Tooltip("Probability", format=".4%")]
    ).properties(
        title=f"{model_name} Class Probabilities"
    ).interactive()
    
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

# --- MODEL LOADING (CACHED) ---

@st.cache_resource
def load_pytorch_model():
    """Loads the trained PyTorch model structure and weights, with download fallback."""
    
    # 1. Check if model exists locally
    if not Path(MODEL_PATH).exists():
        st.error(f"Model file not found: '{MODEL_PATH}'. The Grad-CAM feature is blocked.")
        
        # 2. Attempt to download the model
        if PT_MODEL_DRIVE_URL.endswith("YOUR_PYTORCH_MODEL_ID"):
            st.warning("Please update the `PT_MODEL_DRIVE_URL` in the code with the public link to your PyTorch model file.")
            return None
            
        with st.spinner(f"PyTorch model not found. Attempting to download the large model from Google Drive..."):
            try:
                # Use curl command to download the file directly
                download_command = f"curl -L '{PT_MODEL_DRIVE_URL}' -o '{MODEL_PATH}'"
                exit_code = os.system(download_command)
                
                if exit_code != 0 or not Path(MODEL_PATH).exists() or os.path.getsize(MODEL_PATH) < 100000:
                    raise Exception(f"Download failed or file is corrupt. Exit code: {exit_code}")
                
                st.success(f"PyTorch model successfully downloaded as '{MODEL_PATH}'.")
            except Exception as e:
                st.error(f"Critical Error: Failed to download PyTorch model from Google Drive. Grad-CAM is unavailable. Error: {e}")
                return None


    # 3. Load the model weights
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        st.success("PyTorch model loaded successfully for Grad-CAM.")
        return model
    except Exception as e:
        st.error(f"Error loading model weights from '{MODEL_PATH}'. Check file integrity: {e}")
        return None


pytorch_model = load_pytorch_model()

@st.cache_resource
def load_onnx_model(_model_pt, device):
    """Loads or exports the ONNX model."""
    
    if _model_pt is None:
        st.error("Cannot export ONNX model because the PyTorch model failed to load.")
        return None
        
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
            st.error(f"Failed to export ONNX model: {e}. Check PyTorch model integrity.")
            return None 

    # Load the ONNX runtime session
    try:
        return ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    except Exception as e:
        st.error(f"Failed to initialize ONNX Runtime session: {e}")
        return None

# Load ONNX session only if PyTorch model is ready for export/loading
onnx_session = load_onnx_model(pytorch_model, device)

# --- STREAMLIT UI/LAYOUT ---

# Custom CSS for dark mode aesthetics
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
    
    /* Sidebar styling for better contrast on dark background */
    .css-1d3w5hv { /* This targets the sidebar content area */
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
        st.error("PyTorch Model Failed to Load (Grad-CAM Disabled).")

    if onnx_session:
        st.success("ONNX Runtime Engine Ready.")
    else:
        st.error("ONNX Loading Failed.")
    
    st.markdown("---")
    st.caption("Developed for educational & research purposes.")


# --- CORE APPLICATION LOGIC ---
if uploaded_file:
    
    # --- Data Prep ---
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    
    # --- Classification using the preferred ONNX/PyTorch model ---
    if onnx_session:
        # ONNX Prediction (Used for Final Diagnosis)
        x = transform(img).unsqueeze(0).numpy().astype(np.float32)
        outputs_onnx = onnx_session.run(None, {"input": x})[0][0]
        # Softmax for ONNX output
        probs_onnx = np.exp(outputs_onnx) / np.sum(np.exp(outputs_onnx))
        
        source_model = "ONNX"
        probs = probs_onnx
    
    elif pytorch_model:
        # Fallback to PyTorch prediction if ONNX failed
        img_tensor = transform(img).unsqueeze(0).to(device).to(torch.float32)
        outputs_pt = pytorch_model(img_tensor)
        probs_pt = torch.softmax(outputs_pt, dim=1)[0]
        
        source_model = "PyTorch (Fallback)"
        probs = probs_pt.detach().cpu().numpy()
        st.warning("Using PyTorch model for classification due to ONNX error. Performance may be slower.")

    else:
        st.error("No functioning model available for classification.")
        st.stop()


    # --- Final Diagnosis ---
    final_pred_idx = np.argmax(probs)
    final_diagnosis_class = class_names[final_pred_idx]
    max_prob = probs[final_pred_idx]
    
    
    # --- RESULT SUMMARY & TOP ROW LAYOUT ---
    st.markdown("## 🔬 Diagnosis & Visualization")

    # Use 3 columns for a clean presentation
    col_img, col_heatmap, col_diagnosis = st.columns([1.5, 1.5, 1])

    with col_img:
        st.subheader("Uploaded X-ray")
        st.image(img, use_container_width=True)

    with col_heatmap:
        st.subheader("Model Focus (Grad-CAM)")
        
        if pytorch_model:
            # Grad-CAM requires the PyTorch model
            target_layer = pytorch_model.layer4[-1].conv2 
            
            # Non-normalized tensor for Grad-CAM input
            gradcam_tensor = transform_gradcam(img).unsqueeze(0).to(device).to(torch.float32)
            
            # Check if the model predicted PNEUMONIA (index 1) with the PyTorch model
            # We use the prediction from the PyTorch model for the Grad-CAM logic
            if np.argmax(probs_pt.detach().cpu().numpy()) == 1:
                with st.spinner("Generating Explainability Heatmap..."):
                    heatmap_img = generate_grad_cam(pytorch_model, target_layer, gradcam_tensor, img)
                    st.image(heatmap_img, caption="Areas contributing to diagnosis (Red/Yellow)", use_container_width=True)
            else:
                st.image(img, caption="Grad-CAM visualization (Model decided finding is normal)", use_container_width=True)
                st.info("Grad-CAM is typically most useful for positive findings (e.g., PNEUMONIA).")
        else:
            st.warning("PyTorch model failed to load. Grad-CAM visualization is unavailable.")
            st.image("https://placehold.co/800x600/121212/cf6679?text=Grad-CAM+Unavailable", use_container_width=True)


    with col_diagnosis:
        st.subheader(f"Classification Result ({source_model})")
        
        # Enhanced Result Box
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
        st.caption("Disclaimer: AI models are assistive tools. Clinical interpretation is required.")


    # --- BOTTOM ROW LAYOUT (Charts) ---
    st.markdown("---")
    st.markdown("## 📊 Detailed Prediction Scores")

    col_pt, col_onnx = st.columns(2)

    # PyTorch Chart
    with col_pt:
        st.markdown("### PyTorch Prediction")
        
        if pytorch_model:
            pred_idx_pt = torch.argmax(probs_pt).item()
            pred_class_pt_display = class_names[pred_idx_pt]
            pred_prob_pt = probs_pt[pred_idx_pt].item()
            color_pt = get_status_color(pred_class_pt_display)

            st.markdown(
                f"Highest Confidence: **:{color_pt}[{pred_class_pt_display}]** ({pred_prob_pt*100:.4f}%)"
            )
            df_pt = pd.DataFrame({"Class": class_names, "Probability":[p.item() for p in probs_pt]})
            st.altair_chart(create_conditional_bar_chart(df_pt, "PyTorch"), use_container_width=True) 
        else:
             st.warning("PyTorch model data unavailable for chart.")

    # ONNX Chart
    with col_onnx:
        st.markdown("### ONNX Prediction (Optimized)")
        
        if onnx_session:
            pred_prob_onnx = probs_onnx[final_pred_idx]
            color_onnx = get_status_color(final_diagnosis_class)

            st.markdown(
                f"Highest Confidence: **:{color_onnx}[{final_diagnosis_class}]** ({pred_prob_onnx*100:.4f}%)"
            )
            df_onnx = pd.DataFrame({"Class": class_names, "Probability": probs_onnx.astype(float)})
            st.altair_chart(create_conditional_bar_chart(df_onnx, "ONNX Runtime"), use_container_width=True)
        else:
            st.warning("ONNX model data unavailable for chart.")

# --- NO FILE UPLOADED STATE (Added a placeholder image) ---
else:
    st.info("Upload an X-Ray image in the sidebar to begin the classification process.")
    st.image("https://placehold.co/1000x500/121212/bb86fc?text=Cliniscan+AI+Assistant+%7C+Waiting+for+Image+Upload", use_container_width=True, caption="Sample Chest X-Ray Placeholder")
    






