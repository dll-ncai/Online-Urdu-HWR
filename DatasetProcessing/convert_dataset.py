# %%
import os
import sys
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# %%
DATASETS = r'C:\AliCode\Datasets'
ONLINE_DATASET = 'data_online_line_width'
ONLINE_DATASET_PATH = os.path.join(DATASETS, ONLINE_DATASET)

# %%
import matplotlib
matplotlib.use("Agg")  # IMPORTANT for multiprocessing safety

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict

# %%
df_main = None

files = [1000, 1500, 2000, 2500, 3000, 3500, 5000, 6000]

for csv_file in os.listdir(ONLINE_DATASET_PATH):
    if csv_file.startswith('main') and int(csv_file.split('_')[1].split('.')[0]) in files:
        tmp = pd.read_csv(os.path.join(ONLINE_DATASET_PATH, csv_file))
        if df_main is None:
            df_main = tmp
        else:
            df_main = pd.concat([df_main, tmp])

# %%
feature_values = defaultdict(list)

os.chdir(ONLINE_DATASET_PATH)
df = pd.read_csv('train_leakproof.csv')

for record in tqdm(df.itertuples()):
    csv_data = pd.read_csv(record.csv)

    # Only pen-down
    csv_data = csv_data[csv_data["Pressure"] > 0]

    for col in ["Pressure", "X tilt", "Y tilt"]:
        feature_values[col].append(csv_data[col].values)

# Compute global stats
stats = {}

for col in feature_values:
    values = np.concatenate(feature_values[col])
    low = np.percentile(values, 5)
    high = np.percentile(values, 95)
    mean = values.mean()
    std = values.std()

    stats[col] = {
        "low": low,
        "high": high,
        "mean": mean,
        "std": std
    }

# %%
def tight_crop_img(img_path, thresh_val=250):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, thresh_val, 255, cv2.THRESH_BINARY_INV
    )

    ys, xs = np.where(thresh > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    return img[y1:y2, x1:x2]

# %%
def scale_pressure(values, stats, gamma=1.0):
    mask = values > 0

    low = stats["Pressure"]["low"]
    high = stats["Pressure"]["high"]

    clipped = np.clip(values, low, high)

    scaled = np.zeros_like(values, dtype=float)
    scaled[mask] = (clipped[mask] - low) / (high - low)

    scaled[mask] = scaled[mask] ** gamma

    return scaled


def col_to_linewidth(scaled, min_width=1.5, max_width=7):
    return min_width + scaled * (max_width - min_width)


def scale_tilt(values, stats, feature_name, pressure_mask, alpha=1):
    mask = pressure_mask

    mean = stats[feature_name]["mean"]
    std = stats[feature_name]["std"]

    z = np.zeros_like(values, dtype=float)
    z[mask] = (values[mask] - mean) / std

    scaled = np.zeros_like(values, dtype=float)
    scaled[mask] = np.tanh(alpha * z[mask])

    scaled[mask] = (scaled[mask] + 1) / 2

    return scaled


def scale(col, **kwargs):
    if col == "Pressure":
        return scale_pressure(
            values=kwargs["values"],
            stats=kwargs["stats"],
            gamma=kwargs.get("gamma", 1.0)
        )
    elif col in ["X tilt", "Y tilt"]:
        return scale_tilt(
            values=kwargs["values"],
            stats=kwargs["stats"],
            feature_name=col,
            pressure_mask=kwargs["pressure_mask"],
            alpha=kwargs.get("alpha", 1)
        )
    else:
        raise ValueError(f"Unsupported column: {col}")


def draw_strokes(df, linewidths):
    plt.clf()
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()

    pressure = df['Pressure'].values
    x = df['X cood.'].values
    y = df['Y cood.'].values

    for i in range(len(x) - 1):
        if pressure[i] > 0 and pressure[i + 1] > 0:
            plt.plot(
                x[i:i + 2],
                y[i:i + 2],
                linewidth=linewidths[i],
                color='black'
            )

    plt.axis('off')


# %%
def process_record(csv_path):
    csv_data = pd.read_csv(csv_path)
    csv_data = csv_data.sort_values(by='Time')

    for col in ['Pressure', 'X tilt', 'Y tilt', "Stroke"]:
        if col == "Stroke":
            scaled = np.zeros_like(csv_data["Pressure"].values)
        else:
            scaled = scale(
                col,
                values=csv_data[col].values,
                stats=stats,
                feature_name=col,
                gamma=1.3,
                pressure_mask=csv_data["Pressure"].values > 0,
                alpha=1.3
            )

        draw_strokes(
            csv_data,
            col_to_linewidth(scaled, min_width=0.75, max_width=2.25)
        )

        name = csv_path.split('/')
        name[2] = f'img_{col}'
        os.makedirs('/'.join(name[:-1]), exist_ok=True)
        name = '/'.join(name)[:-4] + '.png'
        name = name.replace('csv', 'img')

        plt.savefig(name, dpi=300, bbox_inches='tight', pad_inches=0)

        cropped = tight_crop_img(name, thresh_val=254)
        cv2.imwrite(name, cropped)

    return 1


# %%
if __name__ == "__main__":
    records = df_main["csv"].tolist()

    with Pool(processes=cpu_count()) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_record, records),
                total=len(records)
            )
        )