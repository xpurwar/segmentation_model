import os
import re
import time
import json
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
)
from monai.networks.nets import SegResNet
from monai.losses import DiceLoss, HausdorffDTLoss
from pathlib import Path

# ================= Configuration =================

data_dir = Path("/mnt/cs/cs153/data/brats_data_purwar/BraTS-MET2025")
output_dir = Path("./autoseg_output")
checkpoint_dir = output_dir / "checkpoints"
test_pred_dir = output_dir / "test_predictions"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
test_pred_dir.mkdir(parents=True, exist_ok=True)

# Load dataset.json config
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

# ================= Device =================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= Transforms =================

# Training transforms including augmentation
train_transforms = Compose([
    # 1. load 4 NIfTIs (as a list) + 1 label
    LoadImaged(keys=["image", "label"]),
    # 3. make sure “label” is (1,H,W,D)  (no_channel→1-length channel)
    EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
    # 4. common spatial/intensity pre-proc + augmentation
    Orientationd(keys=["image","label"], axcodes="RAS"),
    Spacingd(
        keys=["image","label"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear","nearest"),
    ),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image","label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image","label"],
        label_key="label",
        spatial_size=(120,120,120),
        pos=1, neg=1, num_samples=1,
    ),
    RandFlipd(keys=["image","label"], spatial_axis=[0], prob=0.2),
    RandFlipd(keys=["image","label"], spatial_axis=[1], prob=0.2),
    RandFlipd(keys=["image","label"], spatial_axis=[2], prob=0.2),
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
    EnsureTyped(keys=["image","label"]),
])

# Validation/Test transforms: no augmentation, keep full image size
val_test_transforms = Compose([
    LoadImaged(keys=["image"]),   
    EnsureChannelFirstd(keys=["image"]),                       
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.0,1.0,1.0), mode="bilinear"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image"], source_key="image"),
    EnsureTyped(keys=["image"]),
])

# ================= Load Dataset Lists =================

train_datalist = load_decathlon_datalist(
    config["datalist"], is_segmentation=True, data_list_key="training", base_dir=config["dataroot"]
)
val_datalist = load_decathlon_datalist(
    config["datalist"], is_segmentation=True, data_list_key="validation", base_dir=config["dataroot"]
)
test_datalist = load_decathlon_datalist(
    config["datalist"], is_segmentation=True, data_list_key="test", base_dir=config["dataroot"]
)

# ================= Dataset and DataLoader =================
test_ds = CacheDataset(data=test_datalist, transform=val_test_transforms, cache_rate=0.1, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

train_ds = CacheDataset(data=train_datalist, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
imgs, _ = next(iter(train_loader))
print("image batch shape:", imgs.shape)  # should be (B, 4, 120, 120, 120)

val_ds = CacheDataset(data=val_datalist, transform=val_test_transforms, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

print(train_datalist[0]["image"])
# ================= Model Setup =================

model = SegResNet(
    spatial_dims=3,
    in_channels=4,                      # four MRI channels (t1n, t1c, t2w, t2f)
    out_channels=config["output_classes"],  # three output masks
    init_filters=16,
).to(device)

# ================= Loss, Optimizer =================

dice_loss = DiceLoss(softmax=True)
hausdorff_loss = HausdorffDTLoss(softmax=True)

def combined_loss(outputs, targets):
    return dice_loss(outputs, targets) + hausdorff_loss(outputs, targets)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# ================= Label Conversion =================

def convert_labels(raw_label):
    """
    Convert label tensor [B, 1, H, W, D] with labels {0,1,2,3}
    into [B, 3, H, W, D] one-hot masks matching class_names.
    """
    whole_tumor     = ((raw_label == 1) | (raw_label == 2) | (raw_label == 3))
    tumor_core      = ((raw_label == 1) | (raw_label == 3))
    enhancing_tumor = (raw_label == 3)
    targets = torch.cat([
        whole_tumor, tumor_core, enhancing_tumor
    ], dim=1)
    return targets.float()

# ================= Utility: Extract caseID & timepoint =================

def extract_case_timepoint(filename: str):
    """
    Extract 5-digit case ID & 3-digit timepoint from filename.
    E.g. 'BraTS-MET-00598-000.nii.gz' -> ('00598', '000')
    """
    pattern = r"(\d{5})-(\d{3})"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern.")
    return match.group(1), match.group(2)

# ================= Training and Validation =================

# Optional resume checkpoint path (set your checkpoint file here)
resume_checkpoint = None  # or e.g. Path("./autoseg_output/checkpoints/model_epoch_005.pth")
start_epoch = 0
best_val_loss = float("inf")

if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
    print(f"Loading checkpoint from {resume_checkpoint}")
    checkpoint = torch.load(resume_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 1)
    best_val_loss = checkpoint.get("val_loss", float("inf"))
    print(f"Resuming from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
else:
    print("No checkpoint found, training from scratch.")


num_epochs = 700
best_val_loss = float("inf")

print("Starting training...")

start_time = time.time()
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0
    epoch_start = time.time()

    for step, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        raw_labels = batch["label"].to(device)
        targets = convert_labels(raw_labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Batch {step+1}/{len(train_loader)} Loss: {loss.item():.4f}", end="\r")

    avg_train_loss = epoch_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            raw_labels = batch["label"].to(device)
            targets = convert_labels(raw_labels)
            outputs = model(images)
            val_loss += combined_loss(outputs, targets).item()

    avg_val_loss = val_loss / len(val_loader)

    epoch_time = time.time() - epoch_start
    elapsed_time = time.time() - start_time
    epochs_left = num_epochs - (epoch + 1)
    eta_seconds = epochs_left * epoch_time
    eta_mins = eta_seconds / 60

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Validation Loss: {avg_val_loss:.4f}")
    print(f"  Time taken for epoch: {epoch_time:.2f} sec")
    print(f"  Estimated time left: {eta_mins:.2f} min")

    # Save checkpoint each epoch
    chkpt_path = checkpoint_dir / f"model_epoch_{epoch+1:03d}.pth"
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
    }, chkpt_path)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = checkpoint_dir / "best_model.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")

# ================= Test Inference & Save Predictions =================

print("\nStarting inference on test dataset...")

model.eval()

with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(device)  # [1, 4, H, W, D]
        image_meta = batch["image_meta_dict"]
        outputs = model(images)  # [1, 3, H, W, D]
        pred = torch.argmax(outputs, dim=1).cpu().numpy()  # [1, H, W, D]
        pred = pred[0]  # remove batch dimension

        # Extract affine matrix
        affine = image_meta["affine"][0].cpu().numpy()

        # Extract original filename & parse caseID, timepoint
        orig_path = image_meta["filename_or_obj"][0]
        orig_fname = Path(orig_path).name
        case_id, timepoint = extract_case_timepoint(orig_fname)

        # Construct output filename as requested: {caseID}-{timepoint}-seg.nii.gz
        out_fname = f"{case_id}-{timepoint}-seg.nii.gz"
        out_path = test_pred_dir / out_fname

        # Save prediction as uint8 Nifti with input affine and confirmed shape
        pred_nifti = nib.Nifti1Image(pred.astype(np.uint8), affine)
        nib.save(pred_nifti, str(out_path))

        print(f"Saved test prediction: {out_path}")

print("All test predictions saved.")
