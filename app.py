"""
app.py
Gradio web app for Alzheimer's MRI Classification + XAI
(Grad-CAM / LIME / SHAP)

Run:
    python app.py

Requirements:
    pip install gradio lime shap

First-time SHAP background setup (recommended):
    python prepare_shap_background.py
"""

import os
import io
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # must come before pyplot import
import matplotlib.pyplot as plt
import gradio as gr

from model.ensemble_model import FeatureConcatEnsemble
from explainability.grad_cam import GradCAM
from explainability.lime_explain import LIMEExplainer
from explainability.shap_explain import SHAPExplainer

# ── Configuration ──────────────────────────────────────────────────────────────

LABELS = ["CN", "MCI", "AD"]
LABEL_DESC = {
    "CN":  "Cognitively Normal — No signs of Alzheimer's disease.",
    "MCI": "Mild Cognitive Impairment — Early stage; possible precursor to AD.",
    "AD":  "Alzheimer's Disease — Clear signs of neurodegeneration detected.",
}

IMG_SIZE = 256
MODEL_PATH = "outputs/best_model_phase2.pth"
BG_PATH = "outputs/shap_background.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load GPU model (for prediction / Grad-CAM / LIME) ─────────────────────────

print(f"⚙️ Loading main model from {MODEL_PATH} on {device}...")

model = FeatureConcatEnsemble(num_classes=3, freeze_backbones=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device).eval()

print("✅ Main model ready.")

# ── Instantiate explainers using GPU model ─────────────────────────────────────

grad_cam_exp = GradCAM(model, device)
lime_exp = LIMEExplainer(model, device, img_size=IMG_SIZE)

# ── SHAP setup (CPU ONLY to avoid CUDA OOM) ────────────────────────────────────

SHAP_READY = False
shap_exp = None

if os.path.exists(BG_PATH):
    try:
        print("⚙️ Preparing SHAP on CPU...")

        model_cpu = FeatureConcatEnsemble(num_classes=3, freeze_backbones=False)
        model_cpu.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        model_cpu = model_cpu.to("cpu").eval()

        background = torch.load(BG_PATH, map_location="cpu", weights_only=True)

        shap_exp = SHAPExplainer(model_cpu, "cpu", background)
        SHAP_READY = True

        print("✅ SHAP background loaded (CPU mode).")

    except Exception as e:
        SHAP_READY = False
        print(f"⚠️ SHAP initialisation failed: {e}")
else:
    print("⚠️ SHAP background not found. Run: python prepare_shap_background.py")

# ── Preprocessing (must match training loader) ─────────────────────────────────

def preprocess(img_np: np.ndarray):
    """
    Parameters
    ----------
    img_np : (H, W, 3) uint8 RGB image from Gradio

    Returns
    -------
    tensor   : (1, 3, IMG_SIZE, IMG_SIZE) FloatTensor on GPU/CPU
    img_norm : (IMG_SIZE, IMG_SIZE, 3) float32 [0,1] for display
    """
    img = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = img / 255.0

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return tensor, img

# ── Prediction helper ───────────────────────────────────────────────────────────

def predict(img_np: np.ndarray):
    """
    Returns class probability dictionary for Gradio Label component.
    """
    tensor, _ = preprocess(img_np)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

    return {lbl: float(p) for lbl, p in zip(LABELS, probs)}

# ── Matplotlib figure → PIL image ──────────────────────────────────────────────

def fig_to_pil(fig: plt.Figure):
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=130,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )
    buf.seek(0)

    from PIL import Image
    img = Image.open(buf).copy()

    buf.close()
    plt.close(fig)
    return img

# ── Main XAI handler ────────────────────────────────────────────────────────────

def run_xai(img_np: np.ndarray, method: str, lime_samples: int, target_class: str):
    """
    Parameters
    ----------
    img_np       : uploaded MRI image (H, W, 3) uint8
    method       : 'Grad-CAM' | 'LIME' | 'SHAP'
    lime_samples : number of LIME perturbation samples
    target_class : 'Predicted' or one of LABELS

    Returns
    -------
    xai_img   : PIL Image
    prob_dict : dict for Gradio Label
    info_text : Markdown explanation
    """
    if img_np is None:
        return None, {}, "⚠️ Please upload an MRI image first."

    tensor, img_norm = preprocess(img_np)

    # Prediction
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    pred_lbl = LABELS[pred_idx]

    # Resolve target class
    if target_class == "Predicted":
        tgt_idx = pred_idx
    else:
        tgt_idx = LABELS.index(target_class)

    # ── Grad-CAM ───────────────────────────────────────────
    if method == "Grad-CAM":
        cams = grad_cam_exp.generate(tensor, class_idx=tgt_idx)
        fig = grad_cam_exp.visualize(img_norm, cams, labels=LABELS)

    # ── LIME ───────────────────────────────────────────────
    elif method == "LIME":
        explanation = lime_exp.explain(img_norm, num_samples=lime_samples)
        fig = lime_exp.visualize(img_norm, explanation, tgt_idx, labels=LABELS)

    # ── SHAP ───────────────────────────────────────────────
    elif method == "SHAP":
        if not SHAP_READY or shap_exp is None:
            return (
                None,
                {lbl: float(p) for lbl, p in zip(LABELS, probs)},
                "❌ SHAP background not initialized.\n\n"
                "Run:\n```bash\npython prepare_shap_background.py\n```"
            )

        try:
            # move input tensor to CPU for SHAP
            tensor_cpu = tensor.to("cpu")

            shap_values, _ = shap_exp.explain(tensor_cpu, class_idx=tgt_idx)
            fig = shap_exp.visualize(img_norm, shap_values, tgt_idx, labels=LABELS)

        except Exception as e:
            return (
                None,
                {lbl: float(p) for lbl, p in zip(LABELS, probs)},
                f"❌ SHAP failed:\n\n```{str(e)}```"
            )

    else:
        return None, {}, "Unknown method selected."

    xai_img = fig_to_pil(fig)
    prob_dict = {lbl: float(p) for lbl, p in zip(LABELS, probs)}

    device_str = (
        f"CUDA ({torch.cuda.get_device_name(0)})"
        if device.type == "cuda"
        else "CPU"
    )

    info_text = (
        f"### 🧠 Prediction: **{pred_lbl}**\n\n"
        f"{LABEL_DESC[pred_lbl]}\n\n"
        f"| Class | Probability |\n|---|---|\n"
        + "".join(f"| **{lbl}** | `{p:.4f}` |\n" for lbl, p in zip(LABELS, probs))
        + f"\n_XAI method: **{method}** — Target class: **{LABELS[tgt_idx]}** — Device: {device_str}_"
    )

    return xai_img, prob_dict, info_text

# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Alzheimer's MRI — XAI Dashboard",
    theme=gr.themes.Soft(primary_hue="indigo"),
    css=".gradio-container { max-width: 1200px !important; }",
) as demo:

    gr.Markdown(
        """
        # 🧠 Alzheimer's MRI Classification + Explainable AI
        **Ensemble model**: ResNet18 + EfficientNetB0  
        Upload a 2-D MRI brain slice and choose an explainability method.
        """
    )

    with gr.Row():
        # ── Left: Inputs ────────────────────────────────────
        with gr.Column(scale=1):
            img_input = gr.Image(
                label="📤 Upload MRI Slice",
                type="numpy",
                image_mode="RGB",
            )

            method = gr.Radio(
                choices=["Grad-CAM", "LIME", "SHAP"],
                value="Grad-CAM",
                label="🔬 XAI Method",
                info=(
                    "Grad-CAM: fast conv-layer heatmap | "
                    "LIME: superpixel perturbation | "
                    "SHAP: gradient-based attribution (slow, CPU)"
                ),
            )

            target_class = gr.Dropdown(
                choices=["Predicted"] + LABELS,
                value="Predicted",
                label="🎯 Explain for class",
                info="Explain why the model predicts a particular class.",
            )

            lime_samples = gr.Slider(
                minimum=200,
                maximum=2000,
                value=1000,
                step=100,
                label="LIME samples (↑ = slower but more accurate)",
                visible=True,
            )

            run_btn = gr.Button("🔍 Run Explanation", variant="primary", size="lg")

        # ── Right: Outputs ───────────────────────────────────
        with gr.Column(scale=2):
            xai_output = gr.Image(label="🖼️ Explanation Map", type="pil")
            prob_output = gr.Label(label="📊 Class Probabilities", num_top_classes=3)
            info_output = gr.Markdown()

    # Show/hide LIME slider
    method.change(
        fn=lambda m: gr.update(visible=(m == "LIME")),
        inputs=method,
        outputs=lime_samples,
    )

    run_btn.click(
        fn=run_xai,
        inputs=[img_input, method, lime_samples, target_class],
        outputs=[xai_output, prob_output, info_output],
    )

    gr.Markdown(
        """
        ---
        ## 📖 Interpretation Guide

        - **Grad-CAM** → warmer colours (red/yellow) highlight regions that most activated the model.
        - **LIME** → green regions support the prediction; red regions oppose it.
        - **SHAP** → red pixels push the prediction toward the target class; blue pixels push it away.

        > ⚠️ **Note:** SHAP is slower and runs on CPU for memory safety.
        """
    )

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    show_error=True,
)
