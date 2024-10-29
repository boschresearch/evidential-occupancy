"""Export visualizations."""
import matplotlib.pyplot as plt

from scene_reconstruction.visualization.colormap import turbo_black


def image_with_colormap(data, filename, cmap=turbo_black, background=(1.0, 1.0, 1.0)):
    """Export heatmap encoded with colormap as image."""
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig, ax = plt.subplots(1, figsize=(data.shape[1] * px, data.shape[0] * px), facecolor=background)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.savefig(filename, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
