import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path

"""
This script scans the ccHarmony dataset and creates a CSV file mapping each 
composite image to its corresponding real image and frequency-based anomaly 
heatmap. It checks file existence in the specified directories, marks 
whether the pairings are valid, and saves the results as 'ccHarmony_Frequency.csv'.
"""


# —— 0. Directory ——
base_dir    = Path('/content/drive/MyDrive/Final_Project/data/ccHarmony')
comp_dir    = base_dir / 'composite'
real_dir    = base_dir / 'real'
heat_dir    = base_dir / 'freq1'

records = []
for comp_fn in tqdm(os.listdir(comp_dir), desc="scanning composites"):
    name, _ = os.path.splitext(comp_fn)         

    # —— 1. Use rsplit to retain the first two segments as base_id ——
    base_id = name.rsplit('_', 2)[0]            

    # —— 2. real  ——
    real_candidate = None
    for ext in ('.jpg', '.png'):
        if (real_dir / f"{base_id}{ext}").exists():
            real_candidate = f"{base_id}{ext}"
            break
    if real_candidate is None:
        real_candidate = 'Not found'

    # —— 3. heatmap  ——
    heat_candidate = None
    for ext in ('.png', '.jpg'):
        fn = f"{name}_hybrid_anomaly_heatmap{ext}"
        if (heat_dir / fn).exists():
            heat_candidate = fn
            break
    if heat_candidate is None:
        heat_candidate = 'Not found'

    valid = (real_candidate!='Not found') and (heat_candidate!='Not found')

    records.append({
        'composite': comp_fn,
        'real'     : real_candidate,
        'heatmap'  : heat_candidate,
        'valid'    : 'Yes' if valid else 'No'
    })

#—— 4. Save the csv ——
df2    = pd.DataFrame(records)
out_fp = base_dir / 'ccHarmony_Frequency.csv'
df2.to_csv(out_fp, index=False)
print("Done：", out_fp)
