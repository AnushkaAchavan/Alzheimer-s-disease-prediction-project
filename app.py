import base64
import io
import os
import time
import uuid
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explainability.grad_cam import GradCAM
from explainability.lime_explain import LIMEExplainer
from explainability.shap_explain import SHAPExplainer
from model import FeatureConcatEnsemble

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
XAI_DIR = BASE_DIR / "static" / "xai"
WEIGHTS_CANDIDATES = (
    os.getenv("ALZHEIMER_MODEL_WEIGHTS"),
    str(BASE_DIR / "models" / "best_model_phase2.pth"),
    str(BASE_DIR / "outputs" / "best_model_phase2.pth"),
)
SHAP_BACKGROUND_CANDIDATES = (
    os.getenv("ALZHEIMER_SHAP_BACKGROUND"),
    str(BASE_DIR / "models" / "shap_background.pt"),
    str(BASE_DIR / "outputs" / "shap_background.pt"),
)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".nii", ".dcm"}
GENDER_OPTIONS = {"Male", "Female", "Other"}
XAI_METHODS = {"Grad-CAM", "LIME", "SHAP"}
CLASS_KEYS = ("cn", "mci", "ad")
CLASS_LABELS = {
    "cn": "Cognitively Normal",
    "mci": "Mild Cognitive Impairment",
    "ad": "Alzheimer's Disease",
}
DISPLAY_LABELS = ["CN", "MCI", "AD"]
MODEL_IMAGE_SIZE = 256

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-neuroclear-secret")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
XAI_DIR.mkdir(parents=True, exist_ok=True)


class AnalysisError(Exception):
    pass


@dataclass
class ModelAssets:
    runtime: Any
    label_map: dict[str, str]
    model_name: str


LOGIN_USERNAME = os.getenv("NEUROCLEAR_USERNAME", "admin")
LOGIN_PASSWORD = os.getenv("NEUROCLEAR_PASSWORD", "admin123")
REGISTERED_USERS: dict[str, dict[str, str]] = {
    LOGIN_USERNAME: {
        "password": LOGIN_PASSWORD,
        "full_name": "Admin User",
    }
}


class AlzheimerRuntime:
    def __init__(self, weights_path: Path) -> None:
        if torch is None or F is None:
            raise AnalysisError(
                "PyTorch is required to run this model. Install the dependencies in requirements.txt first."
            )

        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(self.device)
        self.grad_cam = GradCAM(self.model, self.device)
        self.lime = LIMEExplainer(self.model, self.device, img_size=MODEL_IMAGE_SIZE)
        self._shap_runtime: tuple[Any, SHAPExplainer] | None = None

    def _load_state_dict(self, map_location: Any) -> Any:
        state_dict = torch.load(self.weights_path, map_location=map_location)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        return strip_module_prefix(state_dict)

    def _build_model(self, device: Any) -> Any:
        model = FeatureConcatEnsemble(num_classes=3, freeze_backbones=False)
        model.load_state_dict(self._load_state_dict(device))
        return model.to(device).eval()

    def _prepare_rgb(self, image_array: np.ndarray) -> np.ndarray:
        image = Image.fromarray((normalize_array(image_array) * 255).astype(np.uint8)).convert("RGB")
        image = image.resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))
        return np.asarray(image, dtype=np.float32) / 255.0

    def _prepare_tensor(self, image_array: np.ndarray, device: Any | None = None) -> Any:
        rgb_array = self._prepare_rgb(image_array)
        tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(device or self.device)

    def predict(self, image_array: np.ndarray) -> dict[str, float]:
        tensor = self._prepare_tensor(image_array)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        return {key: float(probs[index]) for index, key in enumerate(CLASS_KEYS)}

    def _resolve_shap_background_path(self) -> Path | None:
        for candidate in SHAP_BACKGROUND_CANDIDATES:
            if not candidate:
                continue
            path = Path(candidate)
            if path.exists():
                return path
        return None

    def _get_shap_runtime(self) -> tuple[Any, SHAPExplainer]:
        if self._shap_runtime is not None:
            return self._shap_runtime

        background_path = self._resolve_shap_background_path()
        if background_path is None:
            expected_path = BASE_DIR / "models" / "shap_background.pt"
            raise AnalysisError(
                "SHAP background not found. Place 'shap_background.pt' at "
                f"'{expected_path}' or set ALZHEIMER_SHAP_BACKGROUND."
            )

        shap_model = self._build_model(torch.device("cpu"))
        background = torch.load(background_path, map_location="cpu")
        if isinstance(background, dict) and "background" in background:
            background = background["background"]
        if not isinstance(background, torch.Tensor):
            raise AnalysisError("SHAP background file did not contain a tensor.")

        shap_explainer = SHAPExplainer(shap_model, "cpu", background.float())
        self._shap_runtime = (shap_model, shap_explainer)
        return self._shap_runtime

    def explain(
        self,
        image_array: np.ndarray,
        prediction: dict[str, float],
        method: str,
        output_stem: str,
    ) -> dict[str, str | None]:
        rgb_array = self._prepare_rgb(image_array)
        pred_key = max(prediction, key=prediction.get)
        pred_index = CLASS_KEYS.index(pred_key)

        if method == "Grad-CAM":
            tensor = self._prepare_tensor(image_array)
            cams = self.grad_cam.generate(tensor, class_idx=pred_index)
            figure = self.grad_cam.visualize(rgb_array, cams, labels=DISPLAY_LABELS)
            image_url = save_figure_image(output_stem, figure)
            return {
                "image_url": image_url,
                "text": f"Grad-CAM highlights regions that drove the {CLASS_LABELS[pred_key]} prediction.",
            }

        if method == "LIME":
            explanation = self.lime.explain(rgb_array, num_samples=1000)
            figure = self.lime.visualize(rgb_array, explanation, pred_index, labels=DISPLAY_LABELS)
            image_url = save_figure_image(output_stem, figure)
            return {
                "image_url": image_url,
                "text": f"LIME shows superpixel regions that supported or opposed the {CLASS_LABELS[pred_key]} decision.",
            }

        if method == "SHAP":
            _, shap_explainer = self._get_shap_runtime()
            tensor_cpu = self._prepare_tensor(image_array, device=torch.device("cpu"))
            shap_values, resolved_index = shap_explainer.explain(tensor_cpu, class_idx=pred_index)
            figure = shap_explainer.visualize(rgb_array, shap_values, resolved_index, labels=DISPLAY_LABELS)
            image_url = save_figure_image(output_stem, figure)
            resolved_key = CLASS_KEYS[resolved_index]
            return {
                "image_url": image_url,
                "text": f"SHAP visualizes which pixels pushed the model toward or away from {CLASS_LABELS[resolved_key]}.",
            }

        raise AnalysisError(f"Unsupported XAI method: {method}")


class AlzheimerInferenceService:
    def __init__(self) -> None:
        self._assets: ModelAssets | None = None

    def _resolve_weights_path(self) -> Path | None:
        for candidate in WEIGHTS_CANDIDATES:
            if not candidate:
                continue
            path = Path(candidate)
            if path.exists():
                return path
        return None

    def _load_assets(self) -> ModelAssets:
        if self._assets is not None:
            return self._assets

        weights_path = self._resolve_weights_path()
        if weights_path is None:
            expected_path = BASE_DIR / "models" / "best_model_phase2.pth"
            raise AnalysisError(
                "Model weights not found. Place 'best_model_phase2.pth' at "
                f"'{expected_path}' or set ALZHEIMER_MODEL_WEIGHTS."
            )

        runtime = AlzheimerRuntime(weights_path)
        self._assets = ModelAssets(
            runtime=runtime,
            label_map=dict(CLASS_LABELS),
            model_name="FeatureConcatEnsemble (ResNet50 + EfficientNetB0 + DenseNet121)",
        )
        return self._assets

    def analyze(self, file_path: Path, age: int, gender: str, xai_method: str) -> dict[str, Any]:
        started = time.perf_counter()
        assets = self._load_assets()
        image_array = load_mri_array(file_path)
        prediction_output = assets.runtime.predict(image_array)
        result = normalize_prediction(prediction_output, assets.label_map)
        xai = assets.runtime.explain(
            image_array=image_array,
            prediction=prediction_output,
            method=xai_method,
            output_stem=file_path.stem,
        )

        inference_time = round(time.perf_counter() - started, 2)
        return {
            "prediction": result,
            "metadata": {"age": age, "gender": gender},
            "xai": {
                "image_url": xai["image_url"],
                "text": xai["text"] or default_reasoning(result, age, gender),
                "method": xai_method,
            },
            "file": {
                "name": file_path.name,
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                "preview_url": encode_preview(file_path),
            },
            "model_name": assets.model_name,
            "inference_time": inference_time,
            "timestamp": time.strftime("%B %d, %Y %I:%M %p"),
        }


def load_mri_array(file_path: Path) -> np.ndarray:
    suffix = file_path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        image = Image.open(file_path).convert("L").resize((224, 224))
        return np.asarray(image, dtype=np.float32) / 255.0
    if suffix == ".nii":
        if nib is None:
            raise AnalysisError("NIfTI support requires nibabel. Install dependencies first.")
        nii = nib.load(str(file_path))
        volume = np.asarray(nii.get_fdata(), dtype=np.float32)
        return normalize_volume_to_slice(volume)
    if suffix == ".dcm":
        if pydicom is None:
            raise AnalysisError("DICOM support requires pydicom. Install dependencies first.")
        dicom = pydicom.dcmread(str(file_path))
        array = dicom.pixel_array.astype(np.float32)
        if array.ndim == 3:
            array = array[array.shape[0] // 2]
        return resize_to_model_input(normalize_array(array))
    raise AnalysisError(f"Unsupported file type: {suffix}")


def normalize_volume_to_slice(volume: np.ndarray) -> np.ndarray:
    if volume.ndim < 2:
        raise AnalysisError("MRI volume could not be interpreted.")
    if volume.ndim == 2:
        return resize_to_model_input(normalize_array(volume))
    slice_index = volume.shape[-1] // 2
    return resize_to_model_input(normalize_array(volume[..., slice_index]))


def normalize_array(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array)
    min_val = float(array.min())
    max_val = float(array.max())
    if max_val - min_val == 0:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_val) / (max_val - min_val)).astype(np.float32)


def resize_to_model_input(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray((normalize_array(array) * 255).astype(np.uint8))
    image = image.resize((224, 224))
    return np.asarray(image, dtype=np.float32) / 255.0


def strip_module_prefix(state_dict: Any) -> Any:
    if not isinstance(state_dict, dict):
        return state_dict
    return {
        key.replace("module.", "", 1) if isinstance(key, str) and key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def normalize_prediction(prediction_output: Any, label_map: dict[str, str]) -> dict[str, Any]:
    probabilities = parse_probability_mapping(prediction_output)
    predicted_key = max(probabilities, key=probabilities.get)
    predicted_label = label_map.get(predicted_key, predicted_key.upper())
    confidence = round(probabilities[predicted_key] * 100, 1)

    return {
        "predicted_key": predicted_key,
        "predicted_label": predicted_label,
        "diagnosis_code": predicted_key.upper(),
        "confidence": confidence,
        "probabilities": {key: round(value * 100, 1) for key, value in probabilities.items()},
    }


def parse_probability_mapping(prediction_output: Any) -> dict[str, float]:
    if isinstance(prediction_output, dict):
        lowered = {str(key).lower(): float(np.asarray(value).squeeze()) for key, value in prediction_output.items()}
        parsed = {
            "cn": first_number(lowered, ["cn", "healthy", "normal", "class_0"]),
            "mci": first_number(lowered, ["mci", "class_1"]),
            "ad": first_number(lowered, ["ad", "alzheimers", "alzheimer", "class_2"]),
        }
        if parsed["cn"] is not None and parsed["mci"] is not None and parsed["ad"] is not None:
            return normalize_probabilities(parsed)

    array = np.asarray(prediction_output).astype(np.float32).squeeze()
    if array.ndim == 0:
        raise AnalysisError("The model returned a scalar prediction. A 3-class output is expected.")
    if array.size == 3:
        return normalize_probabilities(
            {
                "cn": float(array.flat[0]),
                "mci": float(array.flat[1]),
                "ad": float(array.flat[2]),
            }
        )
    raise AnalysisError("Could not parse prediction output from the model.")


def normalize_probabilities(probabilities: dict[str, float | None]) -> dict[str, float]:
    values = {key: clamp_probability(probabilities.get(key) or 0.0) for key in CLASS_KEYS}
    total = sum(values.values())
    if total <= 0:
        raise AnalysisError("Model probabilities summed to zero.")
    return {key: value / total for key, value in values.items()}


def save_figure_image(output_stem: str, figure: Any) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor=figure.get_facecolor())
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")

    output_name = f"{output_stem}_{uuid.uuid4().hex[:8]}_xai.png"
    output_path = XAI_DIR / output_name
    image.save(output_path)

    buffer.close()
    plt.close(figure)
    return f"/static/xai/{output_name}"


def encode_preview(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        return ""
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    with file_path.open("rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def first_number(mapping: dict[str, Any], candidates: list[str]) -> float | None:
    for candidate in candidates:
        if candidate in mapping:
            return float(np.asarray(mapping[candidate]).squeeze())
    return None


def clamp_probability(value: float) -> float:
    if value > 1:
        value = value / 100.0
    return max(0.0, min(1.0, float(value)))


def default_reasoning(prediction: dict[str, Any], age: int, gender: str) -> str:
    diagnosis = prediction["predicted_label"]
    return (
        f"The model classified this scan as {diagnosis}. "
        f"Age {age} and gender value '{gender}' were included in the request metadata."
    )


service = AlzheimerInferenceService()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("auth", mode="login"))
        return view(*args, **kwargs)

    return wrapped_view


@app.get("/")
def index():
    return render_template(
        "index.html",
        is_logged_in=bool(session.get("logged_in")),
        current_user=session.get("username"),
        current_name=session.get("full_name") or session.get("username"),
    )


@app.route("/auth/<mode>", methods=["GET", "POST"])
def auth(mode: str):
    if mode not in {"login", "register"}:
        return redirect(url_for("auth", mode="login"))

    if session.get("logged_in"):
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        full_name = request.form.get("full_name", "").strip()

        if not username or not password:
            error = "Username and password are required."
        elif mode == "login":
            user = REGISTERED_USERS.get(username)
            if user and user["password"] == password:
                session.clear()
                session["logged_in"] = True
                session["username"] = username
                session["full_name"] = user.get("full_name") or username
                return redirect(url_for("index"))
            error = "Invalid username or password."
        else:
            if not full_name:
                error = "Full name is required to register."
            elif username in REGISTERED_USERS:
                error = "That username is already registered."
            else:
                REGISTERED_USERS[username] = {
                    "password": password,
                    "full_name": full_name,
                }
                session.clear()
                session["logged_in"] = True
                session["username"] = username
                session["full_name"] = full_name
                return redirect(url_for("index"))

    return render_template("auth.html", error=error, mode=mode)


@app.post("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.post("/api/analyze")
@login_required
def analyze():
    try:
        uploaded_file = request.files.get("mri_file")
        age_raw = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()
        xai_method = request.form.get("xai_method", "Grad-CAM").strip()

        if uploaded_file is None or uploaded_file.filename == "":
            raise AnalysisError("Please upload an MRI file before running the analysis.")

        try:
            age = int(age_raw)
        except ValueError as exc:
            raise AnalysisError("Age must be a whole number between 40 and 100.") from exc

        if age < 40 or age > 100:
            raise AnalysisError("Age must be between 40 and 100.")
        if gender not in GENDER_OPTIONS:
            raise AnalysisError("Please choose a valid gender option.")
        if xai_method not in XAI_METHODS:
            raise AnalysisError("Please choose a valid XAI method.")

        suffix = Path(uploaded_file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            raise AnalysisError(f"Unsupported file format. Allowed types: {allowed}.")

        safe_name = f"{uuid.uuid4().hex}{suffix}"
        stored_path = UPLOAD_DIR / safe_name
        uploaded_file.save(stored_path)

        result = service.analyze(stored_path, age, gender, xai_method)
        return jsonify(result)
    except AnalysisError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
