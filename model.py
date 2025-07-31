import json
from pathlib import Path
from monai.apps.auto3dseg import AutoRunner

# Load existing dataset.json
with open("/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025/dataset.json") as f:
    config = json.load(f)

# Extend config with required fields
config["dataroot"] = "/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025"
config["datalist"] = "/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025/dataset.json"
config["modality"] = {str(i): m for i, m in enumerate(["t1n","t1c","t2w","t2f"])}

# Initialize runner
runner = AutoRunner(
    input=config,
    work_dir="autoseg_output",
    train=True,
    analyze=True,
    predict=False,
    infer=False
)

runner.run()
