import os
import cv2
import numpy as np
from scipy.fftpack import fft2, fftshift
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
This script generates anomaly heatmaps for composite images to be used as 
training data for a model. It computes frequency-based and color-based 
anomaly maps using local patch analysis, then combines them into a hybrid 
map (currently using only the color component). The resulting heatmaps are 
saved as .png files for each input image, which can serve as labels or 
auxiliary inputs in model training.
"""

def get_frequency_anomaly_map(comp, patch_size=16, stride=8):
    comp_gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
    h, w = comp_gray.shape
    feats, pos = [], []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = comp_gray[y:y+patch_size, x:x+patch_size]
            log_mag = np.log1p(np.abs(fftshift(fft2(patch))))
            feats.append(np.mean(log_mag))
            pos.append((y, x))
    feats = np.array(feats, np.float32)
    mean_f = feats.mean()
    amap = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for i, (y, x) in enumerate(pos):
        diff = abs(feats[i] - mean_f)
        amap[y:y+patch_size, x:x+patch_size] += diff
        cnt[y:y+patch_size, x:x+patch_size] += 1
    amap /= (cnt + 1e-8)
    return (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

def get_color_anomaly_map(comp, patch_size=16, stride=8):
    lab = cv2.cvtColor(comp, cv2.COLOR_RGB2LAB)
    h, w, _ = lab.shape
    a_vals, b_vals, pos = [], [], []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = lab[y:y+patch_size, x:x+patch_size]
            a_vals.append(patch[...,1].mean())
            b_vals.append(patch[...,2].mean())
            pos.append((y, x))
    a_vals, b_vals = np.array(a_vals, np.float32), np.array(b_vals, np.float32)
    ma, mb = a_vals.mean(), b_vals.mean()
    amap = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for i, (y, x) in enumerate(pos):
        diff = np.hypot(a_vals[i] - ma, b_vals[i] - mb)
        amap[y:y+patch_size, x:x+patch_size] += diff
        cnt[y:y+patch_size, x:x+patch_size] += 1
    amap /= (cnt + 1e-8)
    return (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

# ==== path set ====
composite_dir = "data/selftest/composite"
save_dir = "data/selftest/heatmap_C1"
os.makedirs(save_dir, exist_ok=True)

files = [f for f in os.listdir(composite_dir) if f.lower().endswith(".jpg")]
print(f"Processing {len(files)} composite images...")

for name in tqdm(files):
    comp_img = cv2.cvtColor(cv2.imread(os.path.join(composite_dir, name)), cv2.COLOR_BGR2RGB)

    # addition method
    freq_map = get_frequency_anomaly_map(comp_img)
    color_map = get_color_anomaly_map(comp_img)
    hybrid_map = 0.0 * freq_map + 1.0 * color_map

    base = os.path.splitext(name)[0]
    save_path = os.path.join(save_dir, base + "_hybrid_anomaly_heatmap.png")
    plt.imsave(save_path, hybrid_map, cmap='hot')

print("All done！")
