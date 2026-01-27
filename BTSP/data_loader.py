import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF

class BraTSMRIDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.root_dir = root_dir
        self.samples = []
        self.augment = augment
        self._prepare_samples()
        
    """ 
    It actually iterates through the all the patients and finds the center of the tumor z
    and then the slice numbers that are to be included within the boundaries of the 3d volume
    and append all the slice numbers to the samples 
    """

    def _prepare_samples(self):
        patients = sorted(os.listdir(self.root_dir))

        for patient in patients:
            patient_path = os.path.join(self.root_dir, patient)

            seg_path = os.path.join(
                patient_path,
                [f for f in os.listdir(patient_path) if "seg" in f][0]
            )

            seg = nib.load(seg_path).get_fdata()
            depth = seg.shape[2]

            tumor_slices = [i for i in range(depth) if seg[:, :, i].sum() > 0]
            if len(tumor_slices) == 0:
                continue

            z_min, z_max = min(tumor_slices), max(tumor_slices)
            num_tumor = len(tumor_slices)

            for z in tumor_slices:
                self.samples.append((patient_path, z))

            left_available  = z_min
            right_available = depth - z_max - 1

            left_to_take  = min(num_tumor, left_available)
            right_to_take = min(num_tumor, right_available)

            for i in range(1, left_to_take + 1):
                self.samples.append((patient_path, z_min - i))

            for i in range(1, right_to_take + 1):
                self.samples.append((patient_path, z_max + i))

    def __len__(self):
        return len(self.samples)

    def _normalize_channel(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # simple spatial consistency augmentation so that the model does not get overfitted spatially
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        return image, mask

    def __getitem__(self, idx):
        patient_path, slice_idx = self.samples[idx]

        flair = nib.load(os.path.join(patient_path, [f for f in os.listdir(patient_path) if "flair" in f][0])).get_fdata()
        t1    = nib.load(os.path.join(patient_path, [f for f in os.listdir(patient_path) if "_t1." in f][0])).get_fdata()
        t1ce  = nib.load(os.path.join(patient_path, [f for f in os.listdir(patient_path) if "t1ce" in f][0])).get_fdata()
        t2    = nib.load(os.path.join(patient_path, [f for f in os.listdir(patient_path) if "_t2." in f][0])).get_fdata()
        seg   = nib.load(os.path.join(patient_path, [f for f in os.listdir(patient_path) if "seg" in f][0])).get_fdata()

        flair = torch.tensor(flair[:, :, slice_idx]).unsqueeze(0)
        t1    = torch.tensor(t1[:, :, slice_idx]).unsqueeze(0)
        t1ce  = torch.tensor(t1ce[:, :, slice_idx]).unsqueeze(0)
        t2    = torch.tensor(t2[:, :, slice_idx]).unsqueeze(0)
        seg   = torch.tensor(seg[:, :, slice_idx]).unsqueeze(0)

        # per-channel normalization
        flair = self._normalize_channel(flair)
        t1    = self._normalize_channel(t1)
        t1ce  = self._normalize_channel(t1ce)
        t2    = self._normalize_channel(t2)

        image = torch.cat([flair, t1, t1ce, t2], dim=0)
        mask = (seg > 0).float()

        image = F.interpolate(image.unsqueeze(0), size=(128, 128),
                              mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(128, 128),
                             mode="nearest").squeeze(0)

        if self.augment:
            image, mask = self._augment(image, mask)

        # class-imbalance pixel weight map
        pos = mask.sum()
        neg = mask.numel() - pos
        weight_map = torch.where(mask == 1, neg / (pos + 1e-8), torch.tensor(1.0))

        return image, mask, weight_map
