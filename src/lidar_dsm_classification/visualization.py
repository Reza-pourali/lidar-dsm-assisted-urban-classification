"""Save classification maps and diagnostic figures."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from .classification import CLASS_NAMES


CLASS_COLORS = ["red", "blue", "white", "green", "grey"]


def save_classification_map(labels, output, title):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmap = ListedColormap(CLASS_COLORS)
    patches = [
        Patch(color=color, label=name)
        for color, name in zip(CLASS_COLORS, CLASS_NAMES)
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(labels, cmap=cmap, vmin=1, vmax=5)
    ax.set_title(title)
    ax.axis("off")
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm, output, title):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
