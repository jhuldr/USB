import os
import random
import torch
import numpy as np
import nibabel as nib
from collections import defaultdict


class USBData(torch.utils.data.Dataset):
    """
    USB Dataset Loader
    - Loads healthy and pathology volume pairs
    - Supports random sampling of healthy / pathological volumes
    """

    def __init__(self, data_config, training_: bool = True, device: str = 'cpu'):
        self.args = data_config
        self.training_ = training_
        self.device = device
        
        # Load subject file list
        with open(self.args.data_file, 'r') as f:
            self.data_lists_healthy = [line.strip() for line in f.readlines()]

        self.healthy = self.args.healthy
        self.healthy_proportion = self.args.healthy_proportion

        print("Number of training subjects:", len(self.data_lists_healthy))

    def __len__(self):
        return len(self.data_lists_healthy)

    def __getitem__(self, idx):
        img_path = self.data_lists_healthy[idx]

        img_w_pathol_path = img_path.replace("_healthy.nii", "_pathology.nii")

        fname = os.path.basename(img_path).replace(".nii", "")
        parts = fname.split("_")

        subj_id = parts[0]
        subj_pathol = fname.split("_sample")[0].split("_", 1)[1]

        target = defaultdict(float)

        # load healthy image
        img_healthy = self._load_nib(img_path)
        target["img_wo_pathol"] = img_healthy

        # decide if we use healthy or pathology
        use_healthy = random.random() < self.healthy_proportion

        if self.healthy or use_healthy:
            # use healthy image only
            target["img_w_pathol"] = img_healthy
            target["pathol_name"] = "no_pathology"
            target["pathol_mask"] = torch.zeros_like(img_healthy)

        else:
            # load pathological image
            img_pathol = self._load_nib(img_w_pathol_path)
            target["img_w_pathol"] = img_pathol
            target["pathol_name"] = subj_pathol

            # load pathology mask if available
            mask_path = img_path.replace("_healthy.nii", "_mask.nii")
            if os.path.exists(mask_path):
                mask = torch.tensor(nib.load(mask_path).get_fdata(),
                                    dtype=torch.float).unsqueeze(0)
            else:
                mask = torch.zeros_like(img_healthy)

            target["pathol_mask"] = mask

        target["img_name"] = subj_id
        target = self.convert_floats_to_tensors(target, self.device)

        return target

    def _load_nib(self, filename):
        data = nib.load(filename).get_fdata()
        data = self.normalise(data)
        return torch.tensor(data, dtype=torch.float).unsqueeze(0)

    @staticmethod
    def normalise(data):
        return (data - data.min()) / (data.max() - data.min() + 1e-8)

    @staticmethod
    def convert_floats_to_tensors(data, device):
        """
        Recursively converts all float values in nested structures to tensors.
        """
        if isinstance(data, float):
            return torch.tensor(data, dtype=torch.float, device=device)

        if isinstance(data, dict):
            return {k: USBData.convert_floats_to_tensors(v, device) for k, v in data.items()}

        if isinstance(data, list):
            return [USBData.convert_floats_to_tensors(x, device) for x in data]

        if isinstance(data, tuple):
            return tuple(USBData.convert_floats_to_tensors(x, device) for x in data)

        return data


if __name__ == "__main__":
    pass
