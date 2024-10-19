from openunmix.data import MUSDBDataset
import numpy as np
import tqdm

ds = MUSDBDataset(root="/workspace/musdb18hq", is_wav=True)

for i, (x, y) in enumerate(tqdm.tqdm(ds)):
    d = np.stack([x, y], axis=0)
    np.save(f"/workspace/musdb-np/train/{i}.npy", d)