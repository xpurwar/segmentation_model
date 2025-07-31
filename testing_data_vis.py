import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
print("hello?")
# Load one sample
image_path = Path("/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025/imagesTr/BRATS_00001-000_0000.nii.gz")
label_path = Path("/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025/labelsTr/BRATS_00001-000.nii.gz")

image = nib.load(image_path).get_fdata()
label = nib.load(label_path).get_fdata()

# Pick middle slice
slice_idx = image.shape[2] // 2
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, slice_idx], cmap="gray")
plt.title("Image (t1n or channel 0000)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(label[:, :, slice_idx])
plt.title("Segmentation Mask")
plt.axis("off")

plt.tight_layout()
plt.savefig("sample_visualization.png")
plt.show()