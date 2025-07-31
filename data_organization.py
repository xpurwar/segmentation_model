import shutil
from pathlib import Path

print("crezy?")
# Input data paths
src_dir = Path("/mnt/cs/cs153/data/brats_data_purwar/MICCAI-LH-BraTS2025-MET-Challenge-Training")
#print("Dirs:", [f.name for f in src_dir.iterdir()])
seg_dir = Path("/mnt/cs/cs153/data/brats_data_purwar/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels")

# Desired output location (you specified this directory)
output_root = Path("/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025")
out_images = output_root / "imagesTr"
out_labels = output_root / "labelsTr"
out_images.mkdir(parents=True, exist_ok=True)
out_labels.mkdir(parents=True, exist_ok=True)

# Process each training case
for i, folder in enumerate(sorted(src_dir.glob("BraTS-MET-*"))):
    if not folder.is_dir():
        continue
    if i >= 1:  # Limit to first 20 cases
        break
    case_id = folder.name
    t1n = folder / f"{case_id}-t1n.nii.gz"
    t1c = folder / f"{case_id}-t1c.nii.gz"
    t2w = folder / f"{case_id}-t2w.nii.gz"
    t2f = folder / f"{case_id}-t2f.nii.gz"
    corrected_seg = seg_dir / f"{case_id}-seg.nii.gz"
    default_seg   = folder / f"{case_id}-seg.nii.gz"
    seg = corrected_seg if corrected_seg.exists() else default_seg
    
    print(f"Checking {case_id}")
    print("  Expected files:")
    print(f"    {t1n.exists()=}, {t1n}")
    print(f"    {t1c.exists()=}, {t1c}")
    print(f"    {t2w.exists()=}, {t2w}")
    print(f"    {t2f.exists()=}, {t2f}")
    print(f"    {seg.exists()=}, {seg}")

    if not all(f.exists() for f in [t1n, t1c, t2w, t2f, seg]):
        print(f"Missing file in {case_id}, skipping.")
        continue

    new_id = case_id.replace("BraTS-MET-", "BRATS_")
    shutil.copy(t1n, out_images / f"{new_id}_0000.nii.gz")
    shutil.copy(t1c, out_images / f"{new_id}_0001.nii.gz")
    shutil.copy(t2w, out_images / f"{new_id}_0002.nii.gz")
    shutil.copy(t2f, out_images / f"{new_id}_0003.nii.gz")
    shutil.copy(seg, out_labels / f"{new_id}.nii.gz")

print("✅ Done formatting data.")
