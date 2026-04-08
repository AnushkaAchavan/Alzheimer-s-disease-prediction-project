"""
explainability/lime_explain.py
LIME (Local Interpretable Model-Agnostic Explanations) for FeatureConcatEnsemble.

Install:  pip install lime

Usage:
    from explainability.lime_explain import LIMEExplainer
    exp = LIMEExplainer(model, device)
    explanation = exp.explain(img_np)          # img_np: (H,W,3) float [0,1]
    fig = exp.visualize(img_np, explanation, label_idx=pred_idx)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

try:
    from lime import lime_image
except ImportError:
    raise ImportError("Install LIME: pip install lime")


LABELS = ["CN", "MCI", "AD"]


class LIMEExplainer:
    """
    Wraps LIME's LimeImageExplainer for the MRI ensemble model.

    Notes
    -----
    • Input to the model must be per-image min-max normalised (same as loader.py).
    • LIME internally perturbs superpixel regions, so it works on the raw [0,1] image
      without any additional de-normalisation step.
    • num_samples ≥ 1 000 gives stable results; use 500 for quick demos.
    """

    def __init__(self, model, device, img_size: int = 256):
        self.model    = model
        self.device   = device
        self.img_size = img_size
        self.explainer = lime_image.LimeImageExplainer(
            feature_selection="auto",
            random_state=42,
        )

    # ── Prediction wrapper ─────────────────────────────────────────────────

    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        images : (N, H, W, 3) float64 in [0, 1]  — provided by LIME internally

        Returns
        -------
        (N, num_classes) float32 probability array
        """
        self.model.eval()
        # Convert (N, H, W, 3) → (N, 3, H, W) float32 tensor
        batch = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

        return probs.astype(np.float32)

    # ── Explanation ────────────────────────────────────────────────────────

    def explain(
        self,
        img_np:       np.ndarray,
        num_samples:  int  = 1000,
        num_segments: int  = 50,
    ):
        """
        Parameters
        ----------
        img_np       : (H, W, 3) float32/64 in [0, 1]
        num_samples  : number of perturbed samples LIME generates
        num_segments : number of SLIC superpixels

        Returns
        -------
        lime.lime_image.ImageExplanation
        """
        explanation = self.explainer.explain_instance(
            img_np.astype(np.double),
            self._predict_fn,
            top_labels=3,          # explain all 3 classes
            hide_color=0,          # black-out hidden superpixels
            num_samples=num_samples,
            segmentation_fn=None,  # default: QuickShift
        )
        return explanation

    # ── Visualisation ──────────────────────────────────────────────────────

    def visualize(
        self,
        img_np:      np.ndarray,
        explanation,
        label_idx:   int,
        labels:      list = LABELS,
        num_features: int  = 10,
        save_path:   str  = None,
    ) -> plt.Figure:
        """
        4-panel figure:
          [Original] [Positive regions] [Positive + Negative] [Heatmap]

        Parameters
        ----------
        img_np    : (H, W, 3) float [0, 1]
        label_idx : class to explain (0=CN, 1=MCI, 2=AD)
        save_path : if given, saves the figure
        """
        # ── Positive-only mask ────────────────────────────────────────────
        temp_pos, mask_pos = explanation.get_image_and_mask(
            label_idx,
            positive_only=True,
            num_features=num_features,
            hide_rest=False,
        )

        # ── Positive + Negative mask ──────────────────────────────────────
        temp_both, mask_both = explanation.get_image_and_mask(
            label_idx,
            positive_only=False,
            num_features=num_features,
            hide_rest=False,
        )

        # ── LIME heatmap (continuous weights) ────────────────────────────
        ind = explanation.top_labels[0]
        weights_dict = dict(explanation.local_exp[label_idx])
        segments     = explanation.segments                      # (H, W) int
        heatmap      = np.zeros(segments.shape, dtype=np.float32)
        for seg_id, weight in weights_dict.items():
            heatmap[segments == seg_id] = weight
        # Normalise to [0, 1] for display
        vmax = np.abs(heatmap).max() + 1e-8

        # ── Plot ──────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.patch.set_facecolor("#0f0f0f")

        def _show(ax, img, title, cmap=None):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, color="white", fontsize=9, pad=6)
            ax.axis("off")

        _show(axes[0], img_np, "Original MRI", cmap="gray")
        _show(axes[1], mark_boundaries(temp_pos,  mask_pos),
              f"LIME — {labels[label_idx]}\n(Positive Regions)")
        _show(axes[2], mark_boundaries(temp_both, mask_both),
              f"LIME — {labels[label_idx]}\n(Pos + Neg Regions)")

        # Heatmap overlay
        axes[3].imshow(img_np, cmap="gray", alpha=0.6)
        im = axes[3].imshow(heatmap, cmap="RdBu_r",
                             vmin=-vmax, vmax=vmax, alpha=0.7)
        axes[3].set_title(f"LIME Heatmap\n{labels[label_idx]}", color="white",
                          fontsize=9, pad=6)
        axes[3].axis("off")
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"LIME Explanation  |  Target class: {labels[label_idx]}",
            color="white", fontsize=12, y=1.01,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        return fig
