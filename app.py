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

# --- UI Configuration (MUST BE FIRST) ---
st.set_page_config(
    page_title="Cliniscan X-Ray Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_custom_css():
    """Injects custom CSS for a darker, modern, and cleaner look."""
    st.markdown("""
        <style>
            /* Global Streamlit Overrides */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}

            /* Custom Styling */
            .stApp {
                background-color: #0d1117; /* Dark background */
                color: #c9d1d9; /* Light text */
            }
            
            /* Make titles look professional */
            h1 {
                color: #58a6ff; /* Blue highlight color */
                font-family: 'Inter', sans-serif;
                text-align: center;
                padding-bottom: 10px;
                border-bottom: 2px solid #21262d;
            }
            h3 {
                color: #c9d1d9; 
                border-bottom: 1px solid #21262d;
                padding-bottom: 5px;
                margin-top: 20px;
            }
            h4 {
                color: #58a6ff;
            }

            /* Style the file uploader */
            .stFileUploader {
                border: 2px dashed #30363d;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                transition: background-color 0.3s;
            }
            .stFileUploader > div > label {
                color: #58a6ff;
                font-size: 1.1em;
            }

            /* Style Warning and Info messages */
            div[data-testid="stAlert"] {
                border-left: 5px solid;
                border-radius: 5px;
            }
            
            /* Custom styling for the main result banner */
            .result-banner {
                text-align: center;
                padding: 15px;
                border-radius: 10px;
                margin-top: 10px;
                font-size: 1.5em;
                font-weight: bold;
                background-color: #21262d;
                border: 1px solid #30363d;
            }
            .result-banner.PNEUMONIA {
                color: #f85149; /* Red for warning */
                border-left-color: #f85149;
            }
            .result-banner.NORMAL {
                color: #3fb950; /* Green for safe */
                border-left-color: #3fb950;
            }
            
            /* Style for Streamlit Tabs */
            div[data-baseweb="tab-list"] {
                gap: 15px;
            }
            button[data-baseweb="tab"] {
                background-color: #161b22; /* Darker tab background */
                color: #c9d1d9 !important; 
                border-radius: 8px 8px 0 0;
                padding: 10px 20px;
                border: 1px solid #30363d;
                border-bottom: none;
            }
            button[data-baseweb="tab"][aria-selected="true"] {
                background-color: #21262d; /* Selected tab background */
                color: #58a6ff !important;
                border-bottom: 2px solid #58a6ff;
            }

        </style>
    """, unsafe_allow_html=True)

inject_custom_css()
# --- End UI Configuration ---


# --- Model Configuration ---
device = torch.device("cpu") 

MODEL_PTH_PATH = "model.pth"
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
    target_score.backward(retain_graph=True) 
    
    hook_handle_fm.remove()
    hook_handle_grad.remove()
    
    if not gradients:
        # Return a simple placeholder if Grad-CAM fails
        return Image.new('RGB', (original_img.width, original_img.height), color = 'gray')

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    feature_map = feature_maps[0].squeeze(0)
    for i in range(feature_map.shape[0]):
        feature_map[i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(feature_map, dim=0).relu()
    
    if torch.max(heatmap) > 0:
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
    return "red" if class_name == "PNEUMONIA" else "green"

def create_conditional_bar_chart(df, model_name):
    """Creates a VERTICAL Altair bar chart with conditional colors."""
    
    color_scale = alt.condition(
        alt.datum.Class == "PNEUMONIA",
        alt.value("#f85149"), # Darker red for charts
        alt.value("#3fb950") # Darker green for charts
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
    """Loads the trained PyTorch model structure and weights."""
    
    # --- Check for model file and download if missing ---
    if not os.path.exists(MODEL_PTH_PATH):
        try:
            st.warning("Model weights not found locally. Downloading from Google Drive...")
            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PTH_PATH, quiet=False)
            st.success("Download complete!")
        except Exception as e:
            st.error(f"Failed to download model from Google Drive. Check the file ID and permissions. Error: {e}")
            return None 

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    try:
        model.load_state_dict(torch.load(MODEL_PTH_PATH, map_location=device))
    except Exception as e:
        st.error(f"Failed to load model weights. Error: {e}")
        return None
        
    model.to(device)
    model.eval()
    return model

pytorch_model = load_pytorch_model()

if pytorch_model is None:
    st.stop()


@st.cache_resource
def load_onnx_model(_model_pt, device):
    """Loads or exports the ONNX model."""
    ONNX_PATH = "model.onnx"
    
    if not os.path.exists(ONNX_PATH) or os.path.getsize(ONNX_PATH) < 100000:
        
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
        except Exception as e:
            st.error(f"Failed to export ONNX model: {e}")
            return None 

    try:
        return ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    except Exception as e:
        st.error(f"Failed to load ONNX session: {e}")
        return None


onnx_session = load_onnx_model(pytorch_model, device)

if onnx_session is None:
    st.stop()
# --- End Model Configuration ---


st.title("Cliniscan: Chest X-Ray Pneumonia Classifier") 
st.markdown("A deep learning tool for instant diagnosis and explainability.")


# Center the uploader by using columns
col_left, col_upload, col_right = st.columns([1, 2, 1])

with col_upload:
    uploaded_file = st.file_uploader("Upload Chest X-ray Image (JPG/PNG)", type=["jpg","jpeg","png"])
    if not uploaded_file:
         st.info("Awaiting X-ray upload for analysis...")


if uploaded_file:
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    current_model = pytorch_model 
    target_layer = current_model.layer4[-1].conv2 
    
    img_tensor = transform(img).unsqueeze(0).to(device).to(torch.float32) 
    gradcam_tensor = transform_gradcam(img).unsqueeze(0).to(device).to(torch.float32)
    gradcam_tensor.requires_grad_(True)
    
    
    # --- PYTORCH PREDICTION (for Grad-CAM) ---
    outputs_pt = current_model(img_tensor)
    probs_pt = torch.softmax(outputs_pt, dim=1)[0]
    pred_class_pt = class_names[torch.argmax(probs_pt).item()]

    
    # --- ONNX PREDICTION (Final Result) ---
    x = transform(img).unsqueeze(0).numpy().astype(np.float32)
    outputs_onnx = onnx_session.run(None, {"input": x})[0][0]
    probs_onnx = np.exp(outputs_onnx) / np.sum(np.exp(outputs_onnx))
    
    
    final_pred_idx = np.argmax(probs_onnx)
    final_diagnosis_class = class_names[final_pred_idx]
    max_prob = probs_onnx[final_pred_idx]
    
    st.markdown("---")
    
    # --- DISPLAY FINAL DIAGNOSIS BANNER ---
    if final_diagnosis_class == "PNEUMONIA":
        st.markdown(
            f"<div class='result-banner PNEUMONIA'>⚠️ PNEUMONIA DETECTED | Confidence: {max_prob*100:.2f}%</div>",
            unsafe_allow_html=True
        )
        st.warning("Action Recommended: Please consult a medical professional immediately with this result.")
    else:
        st.markdown(
            f"<div class='result-banner NORMAL'>✅ NORMAL FINDING | Confidence: {max_prob*100:.2f}%</div>",
            unsafe_allow_html=True
        )
        st.info("**Model Analysis:** No Sign of Pneumonia Detected.") 
    st.markdown("---")


    # --- IMPLEMENTING TABS FOR DASHBOARD VIEW ---
    tab_visual, tab_scores = st.tabs(["🖼️ Visual Analysis", "📊 Confidence Scores"])

    with tab_visual:
        st.markdown("### Visualization & Explainability")
        
        col_img, col_gradcam = st.columns(2)
        
        with col_img:
            st.markdown("#### Uploaded X-ray Image")
            st.image(img, use_container_width=True)
            
        with col_gradcam:
            st.markdown("#### Model Focus (Grad-CAM)")
            if pred_class_pt == "PNEUMONIA":
                with st.spinner("Generating Explainability Heatmap..."):
                    heatmap_img = generate_grad_cam(current_model, target_layer, gradcam_tensor, img)
                st.image(heatmap_img, caption="Areas contributing to PNEUMONIA diagnosis (Red/Yellow)", use_container_width=True)
            else:
                st.info("Grad-CAM visualization is typically most useful for positive (PNEUMONIA) cases and is skipped for NORMAL findings.")


    with tab_scores:
        st.markdown("### Deep Learning Model Scores")
        
        col_pt_chart, col_onnx_chart = st.columns(2)

        with col_pt_chart:
            st.markdown("#### PyTorch Prediction")
            pred_idx_pt = torch.argmax(probs_pt).item()
            pred_class_pt = class_names[pred_idx_pt]
            pred_prob_pt = probs_pt[pred_idx_pt].item()
            color_pt = get_status_color(pred_class_pt)

            st.markdown(
                f"Highest Confidence: **:{color_pt}[{pred_class_pt}]** ({pred_prob_pt*100:.4f}%)"
            )
            df_pt = pd.DataFrame({"Class": class_names, "Probability":[p.item() for p in probs_pt]})
            st.altair_chart(create_conditional_bar_chart(df_pt, "PyTorch"), use_container_width=True) 

        with col_onnx_chart:
            st.markdown("#### ONNX Prediction")
            pred_prob_onnx = probs_onnx[final_pred_idx]
            color_onnx = get_status_color(final_diagnosis_class)

            st.markdown(
                f"Highest Confidence: **:{color_onnx}[{final_diagnosis_class}]** ({pred_prob_onnx*100:.4f}%)"
            )
            df_onnx = pd.DataFrame({"Class": class_names, "Probability": probs_onnx})
            st.altair_chart(create_conditional_bar_chart(df_onnx, "ONNX"), use_container_width=True)


