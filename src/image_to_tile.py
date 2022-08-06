import os
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from PIL import Image
from pathlib import Path

data_path = Path(__file__).parents[1] / "data"/ "hubmap-organ-segmentation"
TILE_SIZE = 224

def tile_image(p_img, folder, size: int = 768) -> list:
    w = h = size
    im = np.array(Image.open(p_img))
    # https://stackoverflow.com/a/47581978/4521646
    tiles = [im[i:(i + h), j:(j + w), ...] for i in range(0, im.shape[0], h) for j in range(0, im.shape[1], w)]
    idxs = [(i, (i + h), j, (j + w)) for i in range(0, im.shape[0], h) for j in range(0, im.shape[1], w)]
    name, _ = os.path.splitext(os.path.basename(p_img))
    files = []
    for k, tile in enumerate(tiles):
        if tile.shape[:2] != (h, w):
            tile_ = tile
            tile = np.zeros_like(tiles[0])
            tile[:tile_.shape[0], :tile_.shape[1], ...] = tile_
        p_img = os.path.join(folder, f"{name}_{k:02}.png")
        Image.fromarray(tile).save(p_img)
        files.append(p_img)
    return files, idxs

os.makedirs(data_path / "temp" / "images", exist_ok=True)
os.makedirs(data_path / "temp" / "masks", exist_ok=True)
for dir_source, dir_target in [
    # (os.path.join(str(data_path), 'train_images'), (os.path.join(str(data_path), "temp/images"))),
    (os.path.join(str(data_path), 'train_binary_masks'), (os.path.join(str(data_path), "temp/masks"))),
]:
    ls = glob.glob(os.path.join(dir_source, '*'))
    _= Parallel(n_jobs=3)(
        delayed(tile_image)(p_img, dir_target, size=TILE_SIZE) for p_img in tqdm(ls)
    )

