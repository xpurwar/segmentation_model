import json
from pathlib import Path


# Set the base path
base_dir = Path("/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025")
images_dir = base_dir / "imagesTr"
labels_dir = base_dir / "labelsTr"
modalities = ["t1n", "t1c", "t2w", "t2f"]  # Correspond to _0000 → _0003
num_folds = 5  # or whatever you prefer

# Initialize dataset dictionary
dataset = {
    "modality": {str(i): modality for i, modality in enumerate(modalities)},
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

# Find all base case names
case_ids = sorted({f.name.split("_")[0] + "_" + f.name.split("_")[1] for f in images_dir.glob("*.nii.gz")})

# Populate training list
for i, case_id in enumerate(case_ids):
    image_paths = [
        str(images_dir / f"{case_id}_{str(j).zfill(4)}.nii.gz") for j in range(4)
    ]
    label_path = str(labels_dir / f"{case_id}.nii.gz")

    if not all(Path(p).exists() for p in image_paths) or not Path(label_path).exists():
        print(f"⚠️ Skipping {case_id} due to missing files.")
        continue

    dataset["training"].append({
        "image": image_paths,
        "label": label_path,
        "fold": i % num_folds  # Optional: cross-validation fold info
    })

# Write to file
with open(base_dir / "dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print("✅ dataset.json created at:", base_dir / "dataset.json")
