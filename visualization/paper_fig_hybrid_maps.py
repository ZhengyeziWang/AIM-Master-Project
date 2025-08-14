import os
import cv2
import numpy as np
from scipy.fftpack import fft2, fftshift
import matplotlib.pyplot as plt

"""
This script generates visualization figures for composite images by computing
two types of anomaly heatmaps—frequency anomaly maps and color anomaly maps—
and combining them into a hybrid map. The hybrid map can also be enhanced
for better visualization. The output is a multi-column figure showing the
original composite image, each anomaly map, the hybrid map, and its enhanced
version.
"""

# ---------- Frequency anomaly map calculation ----------
def get_frequency_anomaly_map(comp, patch_size=16, stride=8):
    # Convert to grayscale
    comp_gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
    h, w = comp_gray.shape
    feats, pos = [], []
    # Extract local frequency features from patches
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = comp_gray[y:y+patch_size, x:x+patch_size]
            log_mag = np.log1p(np.abs(fftshift(fft2(patch))))
            feats.append(np.mean(log_mag))
            pos.append((y, x))
    feats = np.array(feats, np.float32)
    mean_f = feats.mean()

    # Aggregate differences from the mean frequency
    amap = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for i, (y, x) in enumerate(pos):
        d = abs(feats[i] - mean_f)
        amap[y:y+patch_size, x:x+patch_size] += d
        cnt[y:y+patch_size, x:x+patch_size] += 1
    amap /= (cnt + 1e-8)

    # Normalize to [0, 1]
    return (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

# ---------- Color anomaly map calculation ----------
def get_color_anomaly_map(comp, patch_size=16, stride=8):
    # Convert to LAB color space
    lab = cv2.cvtColor(comp, cv2.COLOR_RGB2LAB)
    h, w, _ = lab.shape
    a_vals, b_vals, pos = [], [], []
    # Extract average 'a' and 'b' channel values from patches
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = lab[y:y+patch_size, x:x+patch_size]
            a_vals.append(patch[..., 1].mean())
            b_vals.append(patch[..., 2].mean())
            pos.append((y, x))
    a_vals = np.array(a_vals, np.float32)
    b_vals = np.array(b_vals, np.float32)
    ma, mb = a_vals.mean(), b_vals.mean()

    # Aggregate chromaticity differences from the mean
    amap = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for i, (y, x) in enumerate(pos):
        d = np.hypot(a_vals[i] - ma, b_vals[i] - mb)
        amap[y:y+patch_size, x:x+patch_size] += d
        cnt[y:y+patch_size, x:x+patch_size] += 1
    amap /= (cnt + 1e-8)

    # Normalize to [0, 1]
    return (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

# ---------- Visualization enhancement ----------
def enhance_for_visualization(hybrid_map):
    # Convert to uint8
    map_255 = (hybrid_map * 255).astype(np.uint8)
    # Apply histogram equalization
    eq_map = cv2.equalizeHist(map_255)
    # Sharpen with a convolution kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp_map = cv2.filter2D(eq_map, -1, kernel)
    # Normalize back to [0, 1]
    return sharp_map.astype(np.float32) / 255.0

# ---------- Generate figure for paper visualization ----------
def make_hybrid_figure_with_vis(
    composite_dir="data/selftest/comp",
    save_path="figs/hybrid_heatmaps.png",
    max_rows=3,
    alpha=0.5,
    img_exts=(".jpg", ".jpeg", ".png", ".bmp")
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    files = [f for f in os.listdir(composite_dir) if f.lower().endswith(img_exts)]
    files = sorted(files)[:max_rows]
    assert files, "No images found."

    n_rows, n_cols = len(files), 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), dpi=300)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)  # Ensure 2D indexing

    for r, name in enumerate(files):
        path = os.path.join(composite_dir, name)
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Compute maps
        freq = get_frequency_anomaly_map(rgb)
        color = get_color_anomaly_map(rgb)
        hybrid = alpha * freq + (1.0 - alpha) * color
        hybrid = (hybrid - hybrid.min()) / (hybrid.max() - hybrid.min() + 1e-8)

        # Enhanced hybrid map
        hybrid_vis = enhance_for_visualization(hybrid)

        # Plot each column
        col_titles = ["Composite", "FreqMap", "ColorMap", f"Hybrid (α={alpha:.2f})", "Hybrid-VisEnhance"]
        imgs = [rgb, freq, color, hybrid, hybrid_vis]
        cmaps = [None, "hot", "hot", "hot", "hot"]

        for c, (img, title, cmap) in enumerate(zip(imgs, col_titles, cmaps)):
            ax = axes[r, c]
            if cmap:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(img)
            if r == 0:
                ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure to: {save_path}")

if __name__ == "__main__":
    make_hybrid_figure_with_vis(
        composite_dir="data/selftest/comp",
        save_path="figs/hybrid_heatmaps.png",
        max_rows=3,
        alpha=0.5
    )
