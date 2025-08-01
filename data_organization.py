import shutil
import time
from pathlib import Path

print("crezy?")

# Input data paths
src_dir = Path("/mnt/cs/cs153/data/brats_data_purwar/Validation")
seg_dir = Path("/mnt/cs/cs153/data/brats_data_purwar/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels")

# Output paths
output_root = Path("/mnt/cs/cs153/data/brats_data_purwar/Validation")
out_images = output_root / "imagesTr"
out_labels = output_root / "labelsTr"
out_images.mkdir(parents=True, exist_ok=True)
out_labels.mkdir(parents=True, exist_ok=True)

# Gather all cases
cases = sorted([f for f in src_dir.glob("BraTS-MET-*") if f.is_dir()])
total = len(cases)

start_time = time.time()

# Process each training case
for i, folder in enumerate(cases, start=1):
    case_id = folder.name
    t1n = folder / f"{case_id}-t1n.nii.gz"
    t1c = folder / f"{case_id}-t1c.nii.gz"
    t2w = folder / f"{case_id}-t2w.nii.gz"
    t2f = folder / f"{case_id}-t2f.nii.gz"
    corrected_seg = seg_dir / f"{case_id}-seg.nii.gz"
    default_seg   = folder / f"{case_id}-seg.nii.gz"
    seg = corrected_seg if corrected_seg.exists() else default_seg

    if not all(f.exists() for f in [t1n, t1c, t2w, t2f, seg]):
        print(f"❌ Missing file in {case_id}, skipping.")
        continue

    new_id = case_id.replace("BraTS-MET-", "BRATS_")
    shutil.copy(t1n, out_images / f"{new_id}_0000.nii.gz")
    shutil.copy(t1c, out_images / f"{new_id}_0001.nii.gz")
    shutil.copy(t2w, out_images / f"{new_id}_0002.nii.gz")
    shutil.copy(t2f, out_images / f"{new_id}_0003.nii.gz")
    shutil.copy(seg, out_labels / f"{new_id}.nii.gz")

    # ETA estimation
    elapsed = time.time() - start_time
    avg_time = elapsed / i
    remaining = avg_time * (total - i)
    eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
    print(f"✅ Processed {i}/{total} cases – ETA: {eta_str}")

print("🎉 ✅ Done formatting data.")
