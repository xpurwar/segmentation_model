import os
import json
import torch
from pathlib import Path
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ScaleIntensityRanged,
    CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandScaleIntensityd, RandShiftIntensityd,
    EnsureTyped, EnsureType
)
from monai.networks.nets import SegResNet
from monai.losses import DiceLoss, HausdorffDTLoss

# ========== Configuration ==========

data_dir = Path("/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025")
output_dir = Path("./autoseg_output")
os.makedirs(output_dir, exist_ok=True)

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
config["class_names"] = [
    {"name": "whole_tumor",     "index": [1, 2, 3]},
    {"name": "tumor_core",      "index": [1, 3]},
    {"name": "enhancing_tumor", "index": [3]},
]
config["output_classes"] = len(config["class_names"])

# ========== Device ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== MONAI Transforms (edit if needed) ==========
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),  # image [4, H, W, D], label [1, H, W, D]
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "nearest"),
    ),
    ScaleIntensityRanged(
        keys=["image"], a_min=0, a_max=500, b_min=0.0, b_max=1.0, clip=True
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(120, 120, 120),
        pos=1,
        neg=1,
        num_samples=1,
    ),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.2),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.2),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.2),
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
    EnsureTyped(keys=["image", "label"]),
])

# ========== Load Data List ==========
split = "training"  # could be also "validation"/"test" per your setup
datalist = load_decathlon_datalist(
    config["datalist"], is_segmentation=True, data_list_key=split, base_dir=config["dataroot"]
)
print(f"Samples found: {len(datalist)}")

# ========== DataLoader ==========
train_ds = CacheDataset(data=datalist, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

# ========== Model ==========
model = SegResNet(
    spatial_dims=3,
    in_channels=4,                  # Four modalities
    out_channels=config["output_classes"],  # 3 output composite masks
    init_filters=16,
).to(device)

# ========== Loss ==========
dice_loss = DiceLoss(softmax=True)
hausdorff_loss = HausdorffDTLoss(softmax=True)
def combined_loss(outputs, targets):
    return dice_loss(outputs, targets) + hausdorff_loss(outputs, targets)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# ========== Label One-Hot Conversion ==========
def convert_labels(raw_label):
    """
    Converts a label tensor with shape [B, 1, H, W, D] and integer values {0,1,2,3}
    into a [B, 3, H, W, D] tensor according to class_names.
    """
    whole_tumor     = ((raw_label == 1) | (raw_label == 2) | (raw_label == 3))
    tumor_core      = ((raw_label == 1) | (raw_label == 3))
    enhancing_tumor = (raw_label == 3)
    targets = torch.cat([
        whole_tumor, tumor_core, enhancing_tumor
    ], dim=1)
    return targets.float()

# ========== Training Loop ==========
num_epochs = 10

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for step, batch in enumerate(train_loader):
        images = batch["image"].to(device)     # [B, 4, H, W, D]
        raw_labels = batch["label"].to(device) # [B, 1, H, W, D]
        targets = convert_labels(raw_labels)   # -> [B, 3, H, W, D]
        optimizer.zero_grad()
        outputs = model(images)                # [B, 3, H, W, D]
        loss = combined_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Batch {step+1}, Loss: {loss.item():.4f}")
    print(f"===> Epoch [{epoch+1}/{num_epochs}] Average loss: {epoch_loss/len(train_loader):.4f}")

    # Optional: save checkpoint
    torch.save(model.state_dict(), output_dir / f"segresnet_braTS_epoch{epoch+1}.pth")

print("Training complete.")
