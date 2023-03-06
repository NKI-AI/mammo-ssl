from vissl.data.mammo_nki_dataset import MammoNKIDataset
import sys
from typing import Any, List
from vissl.utils.hydra_config import compose_hydra_configuration
from tqdm import tqdm


def hydra_main(overrides: List[Any]):

    cfg = compose_hydra_configuration(overrides)

    ds = MammoNKIDataset(cfg.config, "/projects/mskcc-oncotype/datasets/pretrain/nki-breast-mg-manifest.json", "TRAIN", "nki-mammos", "mammograms")

    for i in tqdm(range(50000)):
        im, success = ds[i]
        im.save(f"/home/j.brunekreef/tmp/mammo-crops/m{i}.png")


if __name__ == "__main__":
    overrides = sys.argv[1:]
    hydra_main(overrides=overrides)
