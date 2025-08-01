from monai.apps.auto3dseg import AutoRunner
from pathlib import Path
import json
from monai.losses import DiceFocalLoss
import torch.nn as nn
from monai.transforms import Lambda

def main():
    data_dir = Path("/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025")
    output_dir = Path("./autoseg_output")

    with open(data_dir / "dataset.json") as f:
        config = json.load(f)

    config["dataroot"] = str(data_dir)
    config["datalist"] = str(data_dir / "dataset.json")
    config["modality"] = "MRI"
    config["dataroot"] = str(data_dir)
    config["datalist"] = str(data_dir / "dataset.json")
    config["modality"] = "MRI"
    config["channel_definitions"] = {
        "t1n": "T1-weighted non-contrast",
        "t1c": "T1-weighted contrast-enhanced",
        "t2w": "T2-weighted",
        "t2f": "FLAIR"
    }
    config["class_names"] = [
    {"name": "whole_tumor",     "index": [1, 2, 3]},
    {"name": "tumor_core",      "index": [1, 3]},
    {"name": "enhancing_tumor", "index": [3]},
    ]
    config["output_classes"] = len(config["class_names"]) 

    config["algorithms"] = [
        {
            "name": "segresnet",
            "modality": "MRI",
            "network": {
                "_target_": "monai.networks.nets.SegResNet",
                "init_filters": 32,
                "blocks_down": [1, 2, 2, 4],
                "in_channels": 4,           # number of input modalities
                "out_channels": 3,          # number of classes
                "norm": "INSTANCE",         # or "BATCH"
                "dropout_prob": 0.2
                # note: SegResNet doesn’t currently accept `use_attn` or `use_conv_attn`
            }
        }
    ]


    runner = AutoRunner(
        input=config,
        work_dir=output_dir,
        algo_path="segresnet",
        train=True,
        analyze=True,
        predict=False,
        infer=False
    )

    runner.run()

if __name__ == "__main__":
    main()
