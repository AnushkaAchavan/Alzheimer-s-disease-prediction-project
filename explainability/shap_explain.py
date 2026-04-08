"""
explainability/shap_explain.py
SHAP GradientExplainer for FeatureConcatEnsemble.

Install:  pip install shap

Usage:
    from explainability.shap_explain import SHAPExplainer
    exp = SHAPExplainer(model, device, background_tensor)
    shap_values, pred_idx = exp.explain(img_tensor)
    fig = exp.visualize(img_np, shap_values, pred_idx)

Background tensor:
    Pass a small batch of representative MRI tensors (shape: (N, 3, H, W)).
    N = 20-100 is usually enough.  You can save one from your val_loader:

        bg = SHAPExplainer.collect_background(val_loader, n=50,
                 save_path="outputs/shap_background.pt")
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    raise ImportError("Install SHAP: pip install shap")


LABELS = ["CN", "MCI", "AD"]


class SHAPExplainer:
    """
    Uses shap.GradientExplainer which is fast, differentiable, and
    works well with multi-backbone PyTorch models.

    Parameters
    ----------
    model      : trained FeatureConcatEnsemble (in eval mode)
    device     : torch.device
    background : (N, 3, H, W) FloatTensor — reference distribution
                 (sample from val_loader or load from disk)
    """

    def __init__(self, model, device, background: torch.Tensor):
        self.model  = model
        self.device = device

        model.eval()
        background = background.to(device).float()
        self.explainer = shap.GradientExplainer(model, background)

    # ── Explanation ────────────────────────────────────────────────────────

    def explain(self, img_tensor: torch.Tensor, class_idx: int = None):
        """
        Parameters
        ----------
        img_tensor : (1, 3, H, W) FloatTensor

        Returns
        -------
        shap_values : list[num_classes] of numpy arrays, each (1, 3, H, W)
        class_idx   : predicted class index
        """
        self.model.eval()
        img_tensor = img_tensor.to(self.device).float()

        if class_idx is None:
            with torch.no_grad():
                logits    = self.model(img_tensor)
                class_idx = int(logits.argmax(dim=1).item())

        shap_values = self.explainer.shap_values(img_tensor)
        return shap_values, class_idx

    # ── Visualisation ──────────────────────────────────────────────────────

    def visualize(
        self,
        img_np:     np.ndarray,
        shap_values,
        class_idx:  int,
        labels:     list = LABELS,
        save_path:  str  = None,
    ) -> plt.Figure:
        """
        One panel per class + original image.

        Red  → pixels pushing the model TOWARD that class
        Blue → pixels pushing the model AWAY from that class
        Gold border marks the predicted class.

        Parameters
        ----------
        img_np    : (H, W, 3) float [0, 1]
        save_path : if given, saves the figure
        """
        num_classes = len(labels)
        fig, axes   = plt.subplots(1, num_classes + 1,
                                   figsize=(5 * (num_classes + 1), 5))
        fig.patch.set_facecolor("#0f0f0f")

        # Original
        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title("Original MRI", color="white", fontsize=10, pad=6)
        axes[0].axis("off")

        # Per-class maps
        for i, label in enumerate(labels):
            sv        = shap_values[i][0]          # (3, H, W)
            sv_summed = sv.sum(axis=0)             # (H, W) sum over channels
            vmax      = np.abs(sv_summed).max() + 1e-8

            ax = axes[i + 1]
            ax.imshow(img_np, cmap="gray", alpha=0.45)
            im = ax.imshow(sv_summed, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, alpha=0.75)

            is_pred       = (i == class_idx)
            border_color  = "gold" if is_pred else "white"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5 if is_pred else 0)

            title = f"SHAP — {label}" + ("\n★ Predicted" if is_pred else "")
            ax.set_title(title, color=border_color, fontsize=9, pad=6)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(
            f"SHAP Gradient Explanation  |  Predicted: {labels[class_idx]}",
            color="white", fontsize=12, y=1.01,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        return fig

    # ── Background helper ──────────────────────────────────────────────────

    @staticmethod
    def collect_background(val_loader, n: int = 50,
                           save_path: str = None) -> torch.Tensor:
        """
        Collect n MRI tensors from val_loader for the SHAP background.

        Example
        -------
            bg = SHAPExplainer.collect_background(
                     val_loader, n=50,
                     save_path="outputs/shap_background.pt")
            exp = SHAPExplainer(model, device, bg)
        """
        collected = []
        for imgs, _ in val_loader:
            collected.append(imgs)
            if sum(b.shape[0] for b in collected) >= n:
                break
        background = torch.cat(collected, dim=0)[:n].cpu().float()

        if save_path:
            torch.save(background, save_path)
            print(f"✅ Saved SHAP background → {save_path}")

        return background
