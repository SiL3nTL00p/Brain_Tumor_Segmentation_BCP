# @title
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
    It actually iterates through all the patients and finds the center of the tumor z
    and then the slice numbers that are to be included within the boundaries of the 3d volume
    and append all the slice numbers to the samples
    """

    def _prepare_samples(self):
        print(f"Checking root directory: {self.root_dir}")
        if not os.path.exists(self.root_dir):
            print(f"Error: Root directory does not exist: {self.root_dir}")
            return

        patient_dirs = []

        # Check if root_dir itself is a patient directory (contains a 'seg' file)
        if any("seg" in f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))):
            patient_dirs.append(self.root_dir)
            print(f"Treating root directory as a single patient folder: {self.root_dir}")
        else:
            # Otherwise, assume root_dir contains multiple patient subdirectories
            items_in_root = sorted(os.listdir(self.root_dir))
            for item in items_in_root:
                full_path = os.path.join(self.root_dir, item)
                if os.path.isdir(full_path):
                    patient_dirs.append(full_path)
            print(f"Found {len(patient_dirs)} patient subdirectories in {self.root_dir}")

        if not patient_dirs:
            print(f"No valid patient directories found in {self.root_dir}")
            return

        for patient_path in patient_dirs:
            try:
                seg_files = [f for f in os.listdir(patient_path) if "seg" in f]
                if not seg_files:
                    print(f"No segmentation file found for patient: {patient_path}. Skipping.")
                    continue
                seg_path = os.path.join(patient_path, seg_files[0])

                seg = nib.load(seg_path).get_fdata()
                depth = seg.shape[2]

                tumor_slices = [i for i in range(depth) if seg[:, :, i].sum() > 0]
                if len(tumor_slices) == 0:
                    print(f"No tumor slices found for patient: {patient_path}. Skipping.")
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
                print(f"Added {len(tumor_slices) + left_to_take + right_to_take} samples for patient {os.path.basename(patient_path)}")
            except Exception as e:
                print(f"Error processing patient {patient_path}: {e}")

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

        # Extract 3D patch centered at slice_idx (depth of 16 slices for proper pooling)
        depth = 16
        z_start = max(0, slice_idx - depth // 2)
        z_end = min(flair.shape[2], z_start + depth)

        # Pad if necessary
        if z_end - z_start < depth:
            z_start = max(0, z_end - depth)

        flair = torch.tensor(flair[:, :, z_start:z_end]).unsqueeze(0).float()
        t1    = torch.tensor(t1[:, :, z_start:z_end]).unsqueeze(0).float()
        t1ce  = torch.tensor(t1ce[:, :, z_start:z_end]).unsqueeze(0).float()
        t2    = torch.tensor(t2[:, :, z_start:z_end]).unsqueeze(0).float()
        seg   = torch.tensor(seg[:, :, z_start:z_end]).unsqueeze(0).float()

        # per-channel normalization
        flair = self._normalize_channel(flair)
        t1    = self._normalize_channel(t1)
        t1ce  = self._normalize_channel(t1ce)
        t2    = self._normalize_channel(t2)

        image = torch.cat([flair, t1, t1ce, t2], dim=0)
        mask = (seg > 0).float()

        # Reduce resolution to save memory
        image = F.interpolate(image.unsqueeze(0), size=(8, 64, 64),
                              mode="trilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(8, 64, 64),
                             mode="nearest").squeeze(0)

        if self.augment:
            image, mask = self._augment(image, mask)


        return image, mask