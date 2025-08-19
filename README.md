# Final Project - Composite Image Harmonization with Anomaly Heatmaps

## 📌 Overview
This project implements an **image harmonization model** trained with auxiliary frequency/color anomaly heatmaps.  
It includes scripts for:
1. **Generating training heatmaps** from composite images
2. **Creating a CSV mapping** between composite, real, and heatmap images
3. **Visualizing hybrid anomaly maps** for paper figures
4. **Main training notebook** (`fullmodel_submit.ipynb`) for model training and evaluation

---

## 📂 Project Structure
```text
Final_Project/
│
├─ fullmodel_submit.ipynb           # Main training notebook (entry point)
│
├─ preprocessing/
│   ├─ generate_training_heatmaps.py  # Generate training anomaly heatmaps
│   ├─ create_ccHarmony_csv.py        # Build composite-real-heatmap CSV mapping
│
├─ visualization/
│   └─ paper_fig_hybrid_maps.py       # Generate paper figure visualizations
│
├─ data/                              # Dataset folder (see below)
│
├─ pretrained weight/                              # (see below)
│
└─ README.md
```

---

## 🖼️ Model Architecture

<div align="center">
  <img src="MP_model.png" alt="Model Structure" width="800"/>
</div>

The model is based on U-Net with a two-level encoder-decoder, followed by global average pooling and a GIFT module to enhance channel-wise attention. A foreground mask is used at inference time to ensure that only the foreground is altered.

---

## 📁 Dataset

The **ccHarmony** dataset contains paired *composite*, *heatmap*, and *real* images for image harmonization tasks.  
Heatmaps are generated **directly from composite images** using frequency and color anomaly analysis, without requiring extra labels or masks.

- 🔗 Dataset link: [ccHarmony (Google Drive)](https://drive.google.com/drive/folders/1Eva_tq4DEfPAlw4Oh5gS0_8jMqmk_gXg?usp=drive_link)

Dataset folder structure:
```text
data/
└─ ccHarmony/
    ├─ composite/
    ├─ real/
    ├─ freq1/                # Folder for generated heatmaps
    └─ ccHarmony_Frequency.csv
```

---
## 📁 Step-by-Step Heatmap Visualization

<div align="center">
  <img src="heatmap_generation.png" alt="Heatmap Visualization" width="800"/>
</div>

This project generates anomaly heatmaps from composite images.  
The figure is arranged in 20 columns, each showing a different step of the pipeline:

1. **Composite** – The original composite image.  
2. **Gray** – The composite converted to grayscale, used as input for frequency analysis.  
3. **PatchGrid** – The sampling grid overlaid on the image (patch size = 16, stride = 8). Each square is a patch used to extract frequency and colour features.  
4. **FreqRaw** – The raw frequency anomaly map, based on deviations of each patch’s frequency feature from the global mean.  
5. **FreqNorm** – The normalized frequency anomaly map (scaled to [0,1]). Redder regions indicate stronger inconsistency.  
6. **SamplePatch** – An example patch (usually from the center) used to illustrate local frequency analysis.  
7. **SampleFFT(log)** – The 2D FFT log-magnitude spectrum of the sample patch, showing its frequency distribution.  
8. **a-chan** – The *a* channel from LAB space (green ↔ red axis).  
9. **b-chan** – The *b* channel from LAB space (blue ↔ yellow axis).  
10. **a-grid(up)** – Patch-wise averages of the *a* channel, upsampled to full resolution.  
11. **b-grid(up)** – Same as above, but for the *b* channel.  
12. **ColorRaw** – The raw colour anomaly map, computed from (a, b) deviations relative to the global mean.  
13. **ColorNorm** – The normalized colour anomaly map (scaled to [0,1]).  
14. **HybridRaw** – The raw hybrid anomaly map, combining frequency and colour anomalies (α = 0.5).  
15. **Overlay-Freq** – The frequency anomaly map overlaid on the composite image.  
16. **Overlay-Color** – The colour anomaly map overlaid on the composite image.  
17. **Overlay-Hybrid** – The hybrid anomaly map overlaid on the composite image.  
18. **EqHist** – The hybrid map after histogram equalization, enhancing global contrast.  
19. **Sharpened** – The equalized map after sharpening, emphasizing local edges.  
20. **Final** – The final enhanced hybrid anomaly map, suitable for publication or presentation.  


---
## 📦 Pretrained Weights

- 📥 [Download trained model weights (Google Drive)](https://drive.google.com/drive/folders/1mtueecc8YBBkZYyT4COflL4NLMNmfCPZ?usp=drive_link)
- File format: `.pth`  
- Saved weights trained for various heatmap(fullmodel_frequency, Color only ,Color-frequency5:5,Color-frequency3:7,Color-frequency7:3,)

---

## 📓 Jupyter Notebook

The core implementation is in:

📘 `fullmodel_submit.ipynb`

---

## 📚 Citation

```bibtex
@inproceedings{niu2023,
title={ccHarmony: Color-Checker Guided Illumination Estimation for Image Harmonization},
author={Niu, Yuge and Zhou, Hong and Huang, Xinxin and Deng, Cheng and Ding, Xuan and Yao, Wei and Dong, Xiaopeng},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
year={2023},
pages={6481--6491}
}
```

🔗 Dataset GitHub: [https://github.com/bcmi/Image-Harmonization-Dataset-ccHarmony](https://github.com/bcmi/Image-Harmonization-Dataset-ccHarmony)
