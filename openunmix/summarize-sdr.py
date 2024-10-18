import json
import math
from pathlib import Path
import musdb
import numpy as np
import tqdm


output_dir = "separate-out/test"

mus = musdb.DB(
    root="../musdb18hq",
    is_wav=True,
    subsets="test",
)

track_median = []

for track in tqdm.tqdm(mus):
    stat_path = Path(output_dir) / (track.name + ".json")
    with open(stat_path, "r") as file:
        stat = json.load(file)

    vocals = next(_ for _ in stat["targets"] if _["name"] == "vocals")
    frame_sdr = [
        _["metrics"]["SDR"]
        for _ in vocals["frames"]
        if not math.isnan(_["metrics"]["SDR"])
    ]
    track_median.append(np.median(frame_sdr))

print("test set median", np.median(track_median))
