from monai.apps.auto3dseg import AutoRunner
from pathlib import Path
import json
from monai.losses import DiceFocalLoss
import torch.nn as nn
from monai.transforms import Lambda
from autoseg_output.segresnet_0.scripts.custom_losses import DiceHausdorffLoss
from monai.networks.nets import SegResNet


def main():
    data_dir = Path("/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025")
    output_dir = Path("./autoseg_output")

    with open(data_dir / "dataset.json") as f:
        config = json.load(f)

    config["dataroot"] = str(data_dir)
    config["datalist"] = str(data_dir / "dataset.json")
    config["modality"] = "MRI"


    config["class_names"] = [
    {"name": "whole_tumor",     "index": [1, 2, 3]},
    {"name": "tumor_core",      "index": [1, 3]},
    {"name": "enhancing_tumor", "index": [3]},
    ]
    config["output_classes"] = len(config["class_names"]) 

    config["sigmoid"] = True

    config["roi_size"] = [96,96,64]
    config["num_steps_per_image"] = 4 

    model = SegResNet(
    spatial_dims=3,
    in_channels=4,               
    out_channels=3,              
    init_filters=8,              
    blocks_down=[1, 2, 2, 4, 4], 
    blocks_up=[1, 1, 1, 1],
    )

    runner = AutoRunner(
        input=config,
        work_dir=output_dir,
        algos="segresnet",
        train=True,
        analyze=True,
        predict=False,
        infer=False, 
        loss=DiceHausdorffLoss(sigmoid=True), 
        device="cuda", 
        ensemble = True
    )

    runner.run()

if __name__ == "__main__":
    main()
