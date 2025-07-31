from monai.apps.auto3dseg import AutoRunner
from pathlib import Path
import json

def main():
    data_dir = Path("/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025")
    output_dir = Path("./autoseg_output")

    with open(data_dir / "dataset.json") as f:
        config = json.load(f)

    config["dataroot"] = str(data_dir)
    config["datalist"] = str(data_dir / "dataset.json")
    config["modality"] = "MRI"
    config["channel_definitions"] = {
    "t1n": "T1-weighted non-contrast",
    "t1c": "T1-weighted contrast-enhanced",
    "t2w": "T2-weighted",
    "t2f": "FLAIR"
    }

    runner = AutoRunner(
        input=config,
        work_dir=output_dir,
        train=True,
        analyze=True,
        predict=False,
        infer=False
    )

    runner.run()

if __name__ == "__main__":
    main()
