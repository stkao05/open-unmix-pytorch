import musdb
from pathlib import Path
import torch
from openunmix import utils
from openunmix import data
from openunmix import predict
from pathlib import Path
import torch
import torchaudio
import json
import numpy as np
import tqdm
import argparse
import museval
import mir_eval

if __name__ == "__main__":
    outdir = "separate-out"
    targets = "vocals"
    model_path = ""
    sample_rate = 44100.0

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using ", device)

    outdir_path = Path(outdir)
    outdir_path.mkdir(exist_ok=True, parents=True)

    mus = musdb.DB(
        root="../musdb18hq",
        is_wav=True,
        subsets="test",
    )
    track = mus.tracks[0]

    separator = utils.load_separator(
        # model_str_or_path="open-unmix",
        targets=["vocals"],
        niter=1,
        wiener_win_len=300,
        device=device,
        pretrained=True,
        residual=True,
        filterbank="torch",
    )
    separator.freeze()
    separator.to(device)

    estimates = predict.separate(
        audio=torch.as_tensor(track.audio.T, dtype=torch.float32),
        rate=sample_rate,
        aggregate_dict=None,
        separator=separator,
        device=device,
    )

    Path(outdir_path / "test").mkdir(exist_ok=True, parents=True)
    for target, estimate in estimates.items():
        if target != "vocals": continue
        target_path = str(outdir_path / "test" / Path(track.name + "-" + target).with_suffix(".wav"))
        torchaudio.save(
            target_path,
            torch.squeeze(estimate).to("cpu"),
            sample_rate=separator.sample_rate,
        )

    eval_est = {
        'vocals': estimates["vocals"].squeeze().T.cpu().numpy(),
        'accompaniment': estimates["residual"].squeeze().T.cpu().numpy(),
    }
    scores = museval.eval_mus_track(track, eval_est, output_dir=outdir_path)
    print(scores)
