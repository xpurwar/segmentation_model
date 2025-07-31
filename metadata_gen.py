import os
import json
from pathlib import Path

# Paths to image and label directories
root = Path("/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025")
imagesTr = root / "imagesTr"
labelsTr = root / "labelsTr"

modalities = ["t1n", "t1c", "t2w", "t2f"]
metadata = {
    "modality": {str(i): mod for i, mod in enumerate(modalities)},
    "labels": {
        "0": "background",
        "1": "NETC",
        "2": "SNFH",
        "3": "ET",
        "4": "RC"
    },
    "training": [],
    "test": []
}

all_ids = sorted(set(f.name.split("_")[0] + "_" + f.name.split("_")[1] for f in imagesTr.glob("*.nii.gz")))
num_folds = 5

for i, case_id in enumerate(all_ids):
    image_paths = [str(imagesTr / f"{case_id}_{i:04d}.nii.gz") for i in range(4)]
    label_path = str(labelsTr / f"{case_id}.nii.gz")

    if not all(Path(p).exists() for p in image_paths + [label_path]):
        print(f"Skipping {case_id} due to missing files.")
        continue

    metadata["training"].append({
        "image": image_paths,
        "label": label_path,
        "fold": i % num_folds
    })

# Write to metadata.json
with open(root / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("✅ metadata.json written to:", root / "metadata.json")
