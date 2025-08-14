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
└─ README.md
```

---

## 🖼️ Model Architecture

<div align="center">
  <img src="mastercalss-.drawio.png" alt="Model Structure" width="800"/>
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

## 📦 Pretrained Weights

- 📥 [Download trained model weights (Google Drive)](https://drive.google.com/drive/folders/1mtueecc8YBBkZYyT4COflL4NLMNmfCPZ?usp=drive_link)
- File format: `.pth`  
- Saved weights trained for 30, 50, and 70 epochs, with decreasing batch size and using perceptual supervision

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
