# %%
from collections import defaultdict
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# %%
DATASETS = 'C:\AliCode\Datasets'
ONLINE_DATASET = 'data_online_line_width_alpha'


ONLINE_DATASET_PATH = os.path.join(DATASETS, ONLINE_DATASET)
os.chdir(ONLINE_DATASET_PATH)

# %%


# %%
df_main = None

files = [1000, 1500, 2000, 2500, 3000, 3500, 5000, 6000]
# files = [1000]
for csv in os.listdir(ONLINE_DATASET_PATH):
    if csv.startswith('main') and int(csv.split('_')[1].split('.')[0]) in files:
        tmp = pd.read_csv(os.path.join(ONLINE_DATASET_PATH, csv))
        if df_main is None:
            df_main = tmp
        else:
            df_main = pd.concat([df_main, tmp])


def tight_crop_img_path(img_path, thresh_val=250):
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


def tight_crop_img(img, thresh_val=250):
    gray = img
    _, thresh = cv2.threshold(
        gray, thresh_val, 255, cv2.THRESH_BINARY_INV
    )

    ys, xs = np.where(thresh > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    return img[y1:y2, x1:x2]

# def scale(df, scaler, gamma=2.0, min_width=1.5, max_width=7):
#     # csv = df[df['Pressure'] > 0]
#     csv = df.copy()
#     csv_scaled = scaler.transform(csv)
#     csv_scaled = pd.DataFrame(csv_scaled, columns=csv.columns)
#     linewidths = min_width + (csv_scaled ** gamma) * (max_width - min_width)

#     return linewidths


def scale_pressure(values, gamma=1.0):
    mask = values > 0

    low = np.percentile(values[mask], 5) if np.any(mask) else 0
    high = np.percentile(values[mask], 95) if np.any(mask) else 1

    clipped = np.clip(values, low, high)

    scaled = np.zeros_like(values, dtype=float)
    scaled[mask] = (clipped[mask] - low) / (high - low)

    # gamma amplification
    scaled[mask] = scaled[mask] ** gamma

    return scaled


def col_to_linewidth(scaled, min_width=1.5, max_width=7):
    return min_width + scaled * (max_width - min_width)


def scale_tilt(values, pressure_mask, alpha=1):
    """
    values: numpy array (full sequence)
    """
    # mask = values != 0  # or use pressure mask if preferred
    mask = pressure_mask

    # mean = stats[feature_name]["mean"]
    # std = stats[feature_name]["std"]

    mean = np.mean(values[mask]) if np.any(mask) else 0
    std = np.std(values[mask]) if np.any(mask) else 1

    z = np.zeros_like(values, dtype=float)
    z[mask] = (values[mask] - mean) / std

    # compress
    scaled = np.zeros_like(values, dtype=float)
    scaled[mask] = np.tanh(alpha * z[mask])

    # map from [-1,1] to [0,1]
    scaled[mask] = (scaled[mask] + 1) / 2

    return scaled


def scale(col, **kwargs):
    if col == "Pressure":
        return scale_pressure(values=kwargs["values"], gamma=kwargs.get("gamma", 1.0))
    # elif col in ["X tilt", "Y tilt"]:
    else:
        return scale_tilt(values=kwargs["values"], pressure_mask=kwargs["pressure_mask"], alpha=kwargs.get("alpha", 1))
    # else:
    #     raise ValueError(f"Unsupported column: {col}")


def draw_strokes(df, linewidths, alpha=None):
    plt.clf()
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()

    pressure = df['Pressure'].values
    x = df['X cood.'].values
    y = df['Y cood.'].values

    for i in range(len(x)-1):
        if pressure[i] > 0 and pressure[i+1] > 0:
            plt.plot(
                x[i:i+2],
                y[i:i+2],
                linewidth=linewidths[i],
                alpha=alpha[i] if alpha is not None else 1.0,
                color='black'
            )

    plt.axis('off')

# %%


def draw_canvas(df, linewidths, alpha=None):

    pressure = df['Pressure'].values
    x = df['X cood.'].values
    y = df['Y cood.'].values

    canvas = np.zeros((y.max() + 1, x.max() + 1), dtype=np.float32)

    for i in range(len(x)-1):
        if pressure[i] > 0 and pressure[i+1] > 0:
            cv2.line(
                canvas,
                (int(x[i]), int(y[i])),
                (int(x[i+1]), int(y[i+1])),
                color=float(alpha[i]) if alpha is not None else 1.0,
                thickness=int(linewidths[i])
            )
    # Invert colors: strokes becomes black (0), background is white (1)
    canvas = 1 - canvas
    canvas = (canvas * 255).clip(0, 255).astype(np.uint8)
    canvas = tight_crop_img(canvas, thresh_val=254)

    return canvas

# %%
# Key based Image Generation


def process_record(csv_path):
    csv = pd.read_csv(csv_path)
    csv = csv.sort_values(by='Time')

    for col in ['Pressure', 'X tilt', 'Y tilt', "Stroke", "dx", "dy", "dt", "vx", "vy", "speed", "theta", "sin_theta", "cos_theta", "dvx", "dvy", "acceleration", "dtheta", "curvature", "stroke_id", "time_norm", "stroke_time", "stroke_duration", "stroke_time_norm"]:
        if col == "Stroke":
            scaled = np.zeros_like(csv["Pressure"].values)
        else:
            scaled = scale(col, values=csv[col].values, gamma=1.3,
                           pressure_mask=csv["Pressure"].values > 0, alpha=1.7)

        # print(scaled.min(), scaled.max())
        if col != "Stroke":
            # draw_strokes(csv, col_to_linewidth(scaled, min_width=0.75, max_width=2.25), col_to_linewidth(scaled, min_width=0.3, max_width=1))
            canvas = draw_canvas(csv, col_to_linewidth(
                scaled, min_width=30, max_width=80), col_to_linewidth(scaled, min_width=0.3, max_width=1))
        else:
            # draw_strokes(csv, col_to_linewidth(scaled, min_width=0.75, max_width=2.25))
            canvas = draw_canvas(csv, col_to_linewidth(
                scaled, min_width=30, max_width=80), )
        # plt.title(f'{col} Visualization', fontsize=16)

        name = csv_path.split('/')
        name[2] = f'img_{col}'
        os.makedirs('/'.join(name[:-1]), exist_ok=True)
        name = '/'.join(name)[:-4] + '.png'
        name = name.replace('csv', 'img')
        # plt.savefig(name, dpi=300, bbox_inches='tight', pad_inches=0)

        # cropped = tight_crop_img(name, thresh_val=254)
        # cv2.imwrite(name, cropped)
        # plt.clf()
        # plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        # plt.title(f'{col} Visualization', fontsize=16)
        # plt.imshow(canvas, cmap='gray')
        # cv2.imwrite(name, canvas)
        canvas = (canvas).clip(0, 255).astype(np.uint8)
        cv2.imwrite(name, canvas)
        # plt.show()


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
