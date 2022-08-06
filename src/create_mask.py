import numpy as np
import os
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path

def rle_decode(mask_rle: str, img_shape: tuple = None) -> np.ndarray:
    seq = mask_rle.split()
    starts = np.array(list(map(int, seq[0::2])))
    lengths = np.array(list(map(int, seq[1::2])))
    assert len(starts) == len(lengths)
    ends = starts + lengths
    img = np.zeros((np.product(img_shape),), dtype=np.uint8)
    for begin, end in zip(starts, ends):
        img[begin:end] = 1
    return img.reshape(img_shape)

data_path = Path(__file__).parents[1] / "data" / "hubmap-organ-segmentation"  
df_train = pd.read_csv(data_path / "train.csv")
LABELS = list(df_train["organ"].unique())
os.makedirs(data_path / "train_binary_masks", exist_ok=True)
os.makedirs(data_path / "train_mclass_masks", exist_ok=True)

for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
    mask = rle_decode(row['rle'], img_shape=(row["img_height"], row["img_width"])).T
    segm_path = data_path / "train_binary_masks" / f"{row['id']}.png"
    Image.fromarray(mask).save(segm_path)
    segm_path = data_path / "train_mclass_masks" / f"{row['id']}.png"
    mask = mask * (LABELS.index(row["organ"]) + 1)
    Image.fromarray(mask).save(segm_path)

