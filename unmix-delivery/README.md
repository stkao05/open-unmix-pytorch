# open-unmix

Setup

```
pip install -r requirements.txt
```

Run source separation script

```
python main.py --outdir [OUT_DIR] --musdb [MUSDB] --wav --subset test
```

The main script accept the following command line argument: 
- `ourdir`: directory where the result of source separation will be saved
- `musdb`: directory of MUSDB18 dataset
- `wav`: whether wav format is used in musdb
- `subset`: split of the dataset ("train" / "test")