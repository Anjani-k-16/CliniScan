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
import gdown 

# --- Configuration ---
# Set page config for a cleaner look (though full black might rely on environment theme)
# This snippet is often used to ensure a dark mode is attempted, but depends on Streamlit version/environment.
# st.set_page_config(layout="wide", initial_sidebar_state="collapsed") 

device = torch.device("cpu") 

# --- Model Artifact Location ---
MODEL_PTH_PATH = "model.pth"
# UNIQUE GOOGLE DRIVE FILE ID:
# This ID is for your model.pth file hosted externally.
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
    """Generates a heat map using Grad-CAM."""
    model.eval()
    # We target the 'PNEUMONIA' class (index 1) for visualization
    target_class_idx = 1 
    
    feature_maps = []
    def forward_hook(module, input, output):
        feature_maps.append(output)
    
    gradients = []
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Register hooks
    hook_handle_fm = target_layer.register_forward_hook(forward_hook)
    hook_handle_grad = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(img_tensor)
    model.zero_grad()
    
    # Backward pass on the target class score
    target_score = output[0, target_class_idx]
    target_score.backward(retain_graph=True) 
    
    # Remove hooks
    hook_handle_fm.remove()
    hook_handle_grad.remove()
    
    # Check if gradients were captured
    if not gradients or len(gradients[0]) == 0:
        # Fallback if gradients are empty (shouldn't happen if setup is correct)
        return Image.new('RGB', (original_img.width, original_img.height), color = 'gray')

    # Compute weights from gradients (Global Average Pooling)
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    feature_map = feature_maps[0].squeeze(0)
    
    # Multiply feature maps by the corresponding gradient weights
    for i in range(feature_map.shape[0]):
        feature_map[i, :, :] *= pooled_gradients[i]
    
    # Sum across channels and apply ReLU to get the heatmap
    heatmap = torch.mean(feature_map, dim=0).relu()
    
    # Normalize the heatmap
    if torch.max(heatmap) > 0:
        heatmap /= torch.max(heatmap)
    
    heatmap_np = heatmap.detach().cpu().numpy() 
    
    # Resize and colorize
    heatmap_resized = cv2.resize(heatmap_np, (original_img.width, original_img.height))
    # Use JET colormap and convert to uint8
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Blend with original image
    img_np = np.array(original_img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 
    
    # Superimpose the heatmap on the original image (50/50 blend)
    superimposed_img = cv2.addWeighted(img_bgr, 0.5, heatmap_colored, 0.5, 0)
    
    # Convert back to PIL Image (RGB format)
    superimposed_img_pil = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    
    return superimposed_img_pil


def get_status_color(class_name):
    """Returns 'red' for PNEUMONIA and 'green' for NORMAL for markdown text."""
    return "red" if class_name == "PNEUMONIA" else "green"

def create_conditional_bar_chart(df, model_name):
    """Creates a VERTICAL Altair bar chart with conditional colors."""
    
    # Define color scale based on the Class value
    color_scale = alt.condition(
        alt.datum.Class == "PNEUMONIA",
        alt.value("red"),
        alt.value("green") 
    )

    chart = alt.Chart(df).mark_bar().encode(
        # X-axis for class, sorted by probability (y-axis)
        x=alt.X("Class", sort="-y", title=None, axis=alt.Axis(labels=True, title=None)), 
        # Y-axis for probability, formatted as percentage
        y=alt.Y("Probability", axis=alt.Axis(format=".0%", title="Probability")),
        color=color_scale,
        tooltip=["Class", alt.Tooltip("Probability", format=".4%")] 
    ).properties(
        title=f"{model_name} Class Probabilities" 
    ).interactive()
    
    return chart



@st.cache_resource
def load_pytorch_model():
    """Loads the trained PyTorch model structure and weights."""
    
    # --- Check for model file and download if missing (NO MESSAGES SHOWN) ---
    if not os.path.exists(MODEL_PTH_PATH):
        with st.spinner(f"Downloading model weights ({MODEL_PTH_PATH})..."):
            try:
                # Download the file silently (quiet=True)
                gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PTH_PATH, quiet=True) 
            except Exception as e:
                st.error(f"Failed to download model from Google Drive. Check the file ID and permissions. Error: {e}")
                return None 
    # ---------------------------------------------------

    # Define the model structure (ResNet18 with 2 output classes)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    try:
        with st.spinner("Loading PyTorch model structure and weights..."):
            # Load the saved state dictionary
            model.load_state_dict(torch.load(MODEL_PTH_PATH, map_location=device))
    except Exception as e:
        st.error(f"Failed to load model weights. Ensure '{MODEL_PTH_PATH}' is a valid PyTorch state dict. Error: {e}")
        return None
        
    model.to(device)
    model.eval()
    return model

# Load PyTorch model (will be silent unless failure occurs)
pytorch_model = load_pytorch_model()

# Handle failure in model loading
if pytorch_model is None:
    # If model loading failed, stop the app immediately
    st.stop()


@st.cache_resource
def load_onnx_model(_model_pt, device):
    """Loads or exports the ONNX model."""
    ONNX_PATH = "model.onnx"
    
    # We only re-export if the file doesn't exist or is too small/corrupted
    if not os.path.exists(ONNX_PATH) or os.path.getsize(ONNX_PATH) < 100000:
        
        # Define a dummy input for the export process
        dummy_input = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)

        try:
            with st.spinner("Exporting PyTorch model to ONNX format (This can take a moment)..."):
                # Export the PyTorch model to ONNX format
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
                # Basic ONNX check
                onnx_model = onnx.load(ONNX_PATH)
                onnx.checker.check_model(onnx_model)
        except Exception as e:
            st.error(f"Failed to export ONNX model: {e}")
            return None 

    # Load ONNX inference session
    try:
        with st.spinner("Loading ONNX inference session..."):
            # Use CPUExecutionProvider for robustness
            return ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    except Exception as e:
        st.error(f"Failed to load ONNX session: {e}")
        return None


# Load ONNX session (will be silent unless failure occurs)
onnx_session = load_onnx_model(pytorch_model, device)

if onnx_session is None:
    st.stop()


# --- Streamlit UI ---

# Custom styling to explicitly try and enforce a dark theme/black background
st.markdown(
    """
    <style>
    /* Main Streamlit container */
    .stApp {
        background-color: #000000;
        color: white; 
    }
    /* Set the main content background to black */
    [data-testid="stVerticalBlock"] {
        background-color: #000000;
    }
    /* Ensure markdown text is visible on dark background */
    h1, h2, h3, h4, p, label {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("CHEST X-RAY CLASSIFIER") 
st.markdown("---")


uploaded_file = st.file_uploader("Choose a Chest X-ray image", type=["jpg","jpeg","png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Prepare tensors for PyTorch and Grad-CAM
    current_model = pytorch_model 
    # Target layer for ResNet18: the last convolutional layer in the last block
    target_layer = current_model.layer4[-1].conv2 
    
    img_tensor = transform(img).unsqueeze(0).to(device).to(torch.float32) 
    gradcam_tensor = transform_gradcam(img).unsqueeze(0).to(device).to(torch.float32)
    gradcam_tensor.requires_grad_(True)
    
    
    # --- PYTORCH PREDICTION ---
    outputs_pt = current_model(img_tensor)
    probs_pt = torch.softmax(outputs_pt, dim=1)[0]
    pred_class_pt = class_names[torch.argmax(probs_pt).item()]

    
    # --- ONNX PREDICTION ---
    # Prepare numpy array for ONNX input
    x = transform(img).unsqueeze(0).numpy().astype(np.float32)
    outputs_onnx = onnx_session.run(None, {"input": x})[0][0]
    # Apply softmax manually for ONNX output (log-softmax is common, so we re-exponentiate)
    probs_onnx = np.exp(outputs_onnx) / np.sum(np.exp(outputs_onnx))
    
    
    # --- FINAL DIAGNOSIS (Using ONNX result) ---
    final_pred_idx = np.argmax(probs_onnx)
    final_diagnosis_class = class_names[final_pred_idx]
    final_diagnosis_color = get_status_color(final_diagnosis_class)
    max_prob = probs_onnx[final_pred_idx]
    
    
    st.markdown("### Classification Result:")
    
    
    if final_diagnosis_class == "PNEUMONIA":
        st.markdown(f"## :{final_diagnosis_color}[PNEUMONIA DETECTED]")
        st.warning("Action Recommended: Please consult a medical professional immediately with this result.")
    else:
        st.markdown(f"## :{final_diagnosis_color}[NORMAL FINDING]")
        
        st.info("**Model Analysis:** No Sign of Pneumonia Detected.") 

    st.markdown(f"Confidence: **{max_prob*100:.2f}%**")
    st.markdown("---")

    st.markdown("### Uploaded X-ray")
    st.image(img, use_container_width=True)
    st.markdown("---")

    
    st.markdown("### Model Focus (Grad-CAM Visualization)")
    # Grad-CAM is only generated if the PyTorch model predicts Pneumonia
    if pred_class_pt == "PNEUMONIA":
        with st.spinner("Generating Explainability Heatmap..."):
            # Use the gradcam_tensor (un-normalized) for better visualization input
            heatmap_img = generate_grad_cam(current_model, target_layer, gradcam_tensor, img)
            st.image(heatmap_img, caption="Areas contributing to PNEUMONIA diagnosis (Red/Yellow)", use_container_width=True)
    else:
        st.info("Grad-CAM visualization skipped as the PyTorch diagnosis is NORMAL.")

    st.markdown("---")

    
    
    st.markdown("### Model Prediction Scores")
    col_pt, col_onnx = st.columns(2)

    # --- PYTORCH CHART ---
    with col_pt:
        st.markdown("#### PyTorch Prediction")
        pred_idx_pt = torch.argmax(probs_pt).item()
        pred_prob_pt = probs_pt[pred_idx_pt].item()
        color_pt = get_status_color(pred_class_pt)

        st.markdown(
            f"Highest Confidence: **:{color_pt}[{pred_class_pt}]** ({pred_prob_pt*100:.4f}%)"
        )
        # Create DataFrame for Altair chart
        df_pt = pd.DataFrame({"Class": class_names, "Probability":[p.item() for p in probs_pt]})
    
        st.altair_chart(create_conditional_bar_chart(df_pt, "PyTorch"), use_container_width=True) 

    # --- ONNX CHART ---
    with col_onnx:
        st.markdown("#### ONNX Prediction")
        pred_prob_onnx = probs_onnx[final_pred_idx]
        color_onnx = get_status_color(final_diagnosis_class)

        st.markdown(
            f"Highest Confidence: **:{color_onnx}[{final_diagnosis_class}]** ({pred_prob_onnx*100:.4f}%)"
        )
        # Create DataFrame for Altair chart
        df_onnx = pd.DataFrame({"Class": class_names, "Probability": probs_onnx})
    
        st.altair_chart(create_conditional_bar_chart(df_onnx, "ONNX"), use_container_width=True)
    














