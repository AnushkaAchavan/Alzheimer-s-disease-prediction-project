"""
explainability/grad_cam.py
Grad-CAM for FeatureConcatEnsemble (ResNet50 + EfficientNetB0 + DenseNet121)

Usage:
    from explainability.grad_cam import GradCAM
    gcam = GradCAM(model, device)
    cams = gcam.generate(img_tensor)           # img_tensor: (1,3,H,W)
    fig  = gcam.visualize(img_np, cams)        # img_np: (H,W,3) float [0,1]
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm


LABELS = ["CN", "MCI", "AD"]


class GradCAM:
    """
    Registers forward/backward hooks on the last conv block of each backbone.

    Target layers chosen:
      • ResNet50      → self.resnet[-2]        (layer4, last residual block)
      • EfficientNetB0→ self.efficientnet[-1]  (last MBConv block)
      • DenseNet121   → self.densenet[-1]      (norm5, after final dense block)
    """

    def __init__(self, model, device):
        self.model  = model
        self.device = device

        self._activations: dict[str, torch.Tensor] = {}
        self._gradients:   dict[str, torch.Tensor] = {}

        self._hooks = []          # keep handles so we can remove if needed
        self._register_hooks()

    # ── Hook registration ──────────────────────────────────────────────────

    def _register_hooks(self):
        targets = {
            "resnet":       self.model.resnet[-2],          # layer4
            "efficientnet": self.model.efficientnet[-1],    # last MBConv
            "densenet":     self.model.densenet[-1],        # norm5
        }

        for name, layer in targets.items():
            self._hooks.append(
                layer.register_forward_hook(self._make_fwd_hook(name))
            )
            self._hooks.append(
                layer.register_full_backward_hook(self._make_bwd_hook(name))
            )

    def _make_fwd_hook(self, name):
        def hook(module, inp, out):
            self._activations[name] = out.detach()
        return hook

    def _make_bwd_hook(self, name):
        def hook(module, grad_in, grad_out):
            self._gradients[name] = grad_out[0].detach()
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    # ── Core generation ────────────────────────────────────────────────────

    def generate(self, img_tensor: torch.Tensor, class_idx: int = None) -> dict:
        """
        Parameters
        ----------
        img_tensor : (1, 3, H, W) FloatTensor  – already on any device
        class_idx  : target class (None → use predicted class)

        Returns
        -------
        dict with keys: 'resnet', 'efficientnet', 'densenet', 'fused',
                        'class_idx', 'probs'
        """
        self.model.eval()
        img_tensor = img_tensor.to(self.device).float()

        # ── Forward ──────────────────────────────────────────────────────
        img_tensor.requires_grad_(True)
        output = self.model(img_tensor)                    # (1, num_classes)
        probs  = torch.softmax(output.detach(), dim=1).cpu().numpy()[0]

        if class_idx is None:
            class_idx = int(probs.argmax())

        # ── Backward ─────────────────────────────────────────────────────
        self.model.zero_grad()
        output[0, class_idx].backward()

        # ── Compute per-backbone CAMs ─────────────────────────────────────
        cams: dict = {}
        for name in ("resnet", "efficientnet", "densenet"):
            grads = self._gradients[name]       # (1, C, h, w)
            acts  = self._activations[name]     # (1, C, h, w)

            # Global-average-pool the gradients → channel weights
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam     = (weights * acts).sum(dim=1).squeeze()  # (h, w)
            cam     = F.relu(cam).cpu().numpy()

            # Normalise to [0, 1]
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            cams[name] = cam

        # ── Fused CAM (resize each to 256×256, mean) ─────────────────────
        H = W = img_tensor.shape[-1]
        fused = np.zeros((H, W), dtype=np.float32)
        for cam in cams.values():
            fused += cv2.resize(cam, (W, H))
        fused /= 3
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

        cams["fused"]     = fused
        cams["class_idx"] = class_idx
        cams["probs"]     = probs

        return cams

    # ── Overlay helper ─────────────────────────────────────────────────────

    @staticmethod
    def overlay(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """
        Blend a JET-coloured heatmap over the original MRI slice.

        Parameters
        ----------
        img_np : (H, W, 3) float [0, 1]
        cam    : (H, W)    float [0, 1]

        Returns
        -------
        (H, W, 3) float [0, 1]
        """
        H, W = img_np.shape[:2]
        cam_resized = cv2.resize(cam, (W, H))

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        blended = alpha * heatmap + (1 - alpha) * img_np
        return np.clip(blended, 0, 1)

    # ── Visualisation ──────────────────────────────────────────────────────

    def visualize(
        self,
        img_np: np.ndarray,
        cams:   dict,
        labels: list = LABELS,
        save_path: str = None,
    ) -> plt.Figure:
        """
        Creates a 5-panel figure:
          [Original] [ResNet CAM] [EfficientNet CAM] [DenseNet CAM] [Fused CAM]

        Parameters
        ----------
        img_np    : (H, W, 3) float [0, 1]
        cams      : dict returned by generate()
        save_path : if given, saves the figure

        Returns
        -------
        matplotlib Figure
        """
        backbone_keys   = ["resnet", "efficientnet", "densenet", "fused"]
        backbone_labels = ["ResNet50", "EfficientNetB0", "DenseNet121", "Fused"]

        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        fig.patch.set_facecolor("#0f0f0f")

        # Original
        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title("Original MRI", color="white", fontsize=10, pad=6)
        axes[0].axis("off")

        # Per-backbone + fused
        for ax, key, name in zip(axes[1:], backbone_keys, backbone_labels):
            overlaid = self.overlay(img_np, cams[key])
            ax.imshow(overlaid)
            ax.set_title(f"Grad-CAM\n{name}", color="white", fontsize=9, pad=6)
            ax.axis("off")

        probs     = cams["probs"]
        pred_lbl  = labels[cams["class_idx"]]
        prob_str  = "   ".join(f"{l}: {p:.2f}" for l, p in zip(labels, probs))
        fig.suptitle(
            f"Predicted: {pred_lbl}   |   {prob_str}",
            color="white", fontsize=12, y=1.01,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        return fig