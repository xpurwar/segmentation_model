import json
from pathlib import Path
from collections import defaultdict
import re


# Set up base directories
base_dir = Path("/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025")
images_dir = base_dir / "imagesTr"
labels_dir = base_dir / "labelsTr"
modalities = ["t1n", "t1c", "t2w", "t2f"]  # Correspond to _0000 → _0003
val_dir = Path("/mnt/cs/cs153/data/brats_data_purwar/Validation")
num_folds = 5

# Initialize dataset dict
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
    "validation": [],
    "test": []
}

# --- Step 1: Handle training/validation split (from labeled data) ---
case_ids = sorted({f.name.split("_")[0] + "_" + f.name.split("_")[1] for f in images_dir.glob("*.nii.gz")})
split_idx = int(0.8 * len(case_ids))  # 80% train, 20% validation

for i, case_id in enumerate(case_ids):
    image_paths = [
        str(images_dir / f"{case_id}_{str(j).zfill(4)}.nii.gz") for j in range(4)
    ]
    label_path = str(labels_dir / f"{case_id}.nii.gz")

    if not all(Path(p).exists() for p in image_paths) or not Path(label_path).exists():
        print(f"⚠️ Skipping {case_id} due to missing files.")
        continue

    entry = {
        "image": image_paths,
        "label": label_path,
        "fold": i % num_folds
    }

    if i < split_idx:
        dataset["training"].append(entry)
    else:
        dataset["validation"].append(entry)

# --- Step 2: Handle test split (unlabeled validation data using scan-type suffixes) ---

val_cases = defaultdict(dict)

# Required modality order
modality_order = ["t1n", "t1c", "t2w", "t2f"]

print("\n🔍 Scanning validation directory for test cases...")

for file in val_dir.rglob("*.nii.gz"):
    fname = file.name.replace(".nii.gz", "")  # strip extension
    if not fname.count("-") >= 2:
        print(f"⚠️ Skipping malformed filename: {file.name}")
        continue

    # Split and re-join to isolate modality
    base, modality = fname.rsplit("-", 1)  # e.g., 'BraTS-MET-00881-000', 't1c'
    case_id = base  # full ID including timepoint

    if modality not in modality_order:
        print(f"⚠️ Unknown modality {modality} in file: {file.name}")
        continue

    print(f"Found file: {file.name} → Case ID: {case_id}, Modality: {modality}")
    val_cases[case_id][modality] = str(file)


print("\n Checking if all modalities are present per case:")
for case_id, paths in val_cases.items():
    found_modalities = list(paths.keys())
    print(f" {case_id} → Found: {found_modalities}")

    if all(m in paths for m in modality_order):
        image_paths = [paths[m] for m in modality_order]
        dataset["test"].append({
            "image": image_paths,
            "label": None
        })
        print(f" Added {case_id} to test set.")
    else:
        missing = [m for m in modality_order if m not in paths]
        print(f" Skipping {case_id} — missing modalities: {missing}")


# --- Step 3: Write to file ---
json_path = base_dir / "dataset.json"
with open(json_path, "w") as f:
    json.dump(dataset, f, indent=4)

# --- Step 4: Print stats ---
print(f"✅ dataset.json created at: {json_path}")
print(f"Training cases:   {len(dataset['training'])}")
print(f"Validation cases: {len(dataset['validation'])}")
print(f"Test cases:       {len(dataset['test'])}")
