import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class BraTSMRIDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self._prepare_samples()

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

            z_min = min(tumor_slices)
            z_max = max(tumor_slices)
            num_tumor = len(tumor_slices)

            # add all tumor slices
            for z in tumor_slices:
                self.samples.append((patient_path, z))

            # compute available non-tumor range safely
            left_available  = z_min
            right_available = depth - z_max - 1

            left_to_take  = min(num_tumor, left_available)
            right_to_take = min(num_tumor, right_available)

            # add left non-tumor slices
            for i in range(1, left_to_take + 1):
                self.samples.append((patient_path, z_min - i))

            # add right non-tumor slices
            for i in range(1, right_to_take + 1):
                self.samples.append((patient_path, z_max + i))

    def __len__(self):
        return len(self.samples)

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

        image = torch.cat([flair, t1, t1ce, t2], dim=0)
        mask = (seg > 0).float()

        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        image = F.interpolate(
            image.unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(128, 128),
            mode="nearest"
        ).squeeze(0)

        return image, mask
