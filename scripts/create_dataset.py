"""
Reconstruct images and plot compared to original images.
"""
import os
import torch
import argparse
import nibabel as nib
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import get_dataset


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#SET ALL SEEDS
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

def rescale_tensor(tensor, min_val=0, max_val=255):
    return tensor * (max_val - min_val) + min_val

def convert_to_nifti(img, aff=None):
    img = img.numpy()
    img = np.squeeze(img)
    if aff is None:
        img = nib.Nifti1Image(img, affine=np.eye(4))
    else:
        img = nib.Nifti1Image(img, affine=aff)
    return img

def get_unique_filename(out_name, path):
    existing_files = set(os.listdir(path))  
    unique_name = out_name
    counter = 2

    while f"{unique_name}_healthy.nii" in existing_files or f"{unique_name}_healthy.nii.gz" in existing_files:
        unique_name = f"{out_name}{counter}"
        counter += 1

    return unique_name

def save_sample_paths(file_path, sample_paths):
    with open(file_path, 'w') as f:
        for path in sample_paths:
            f.write(path + '\n')


def main(args):
    # Setup PyTorch:
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    args.training = True
    args.dataset = 'una'
    dataset = get_dataset(args, device=device)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )   
    
    train_save_folder = os.path.join(args.save_path, 'train_samples')
    os.makedirs(train_save_folder, exist_ok=True)


    args.data_config_path = args.data_config_path.replace('train', 'test')
    args.training = False
    args.dataset = 'una'
    dataset = get_dataset(args, device=device)

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )   
    
    test_save_folder = os.path.join(args.save_path, 'test_samples')
    os.makedirs(test_save_folder, exist_ok=True)

    paths = [train_save_folder, test_save_folder]
    loaders = [test_loader]

    sample_range = 1
    sample_paths = {
        'train': {'healthy': [], 'pathology': [], 'mask': []},
        'test': {'healthy': [], 'pathology': [], 'mask': []}
    }

    for path, loader, dataset_type in zip(paths, loaders, ['train', 'test']):
        print(f"making samples for path: {path}")

        for samp_step in range(sample_range):
            print(f"sample step {samp_step}")
            for step, batch in tqdm(enumerate(loader), total=len(loader)):

                image_name = batch['name']

                sample_im = batch['input_healthy']
                pathologies = batch['input_pathol']
                patholprob = batch['pathology']
                pathology_name = batch['pathology_file']
                T1_affine = batch['T1_affine']
                pathology_name = [name.split('/')[-1] for name in pathology_name]
                pathology_name = [name.split('.nii')[0] for name in pathology_name]
                
                for i, image in enumerate(sample_im):
                    
                    image_name_ = image_name[i]

                    skip_flag = False
                    for fname in os.listdir(path):
                        if image_name_ in fname:
                            print(f"File {fname} exist, skip!")
                            skip_flag = True
                            break
                    if skip_flag:
                        continue
                    
                    pathology_name_ = pathology_name[i]
                    aff = T1_affine[i]
                    
                    # Convert images to NIfTI format
                    image_nifti = convert_to_nifti(image, aff)
                    pathol_nifti = convert_to_nifti(pathologies[i], aff)
                    patholprob_nifti = convert_to_nifti(patholprob[i], aff)

                    # Generate unique output name for files
                    out_name = f"{image_name_}_{pathology_name_}_sample"
                    unique_out_name = get_unique_filename(out_name, path)
                    healthy_out = unique_out_name + '_healthy'
                    pathol_out = unique_out_name + '_pathology'
                    patholprob_out = unique_out_name + '_mask'

                    # Save the images as NIfTI files
                    healthy_path = os.path.join(path, f"{healthy_out}.nii")
                    pathology_path = os.path.join(path, f"{pathol_out}.nii")
                    pathologybin_path = os.path.join(path, f"{patholprob_out}.nii")
                    nib.save(image_nifti, healthy_path)
                    nib.save(pathol_nifti, pathology_path)
                    nib.save(patholprob_nifti, pathologybin_path)

                    # Append the file paths to the dictionary based on dataset type
                    sample_paths[dataset_type]['healthy'].append(healthy_path)
                    sample_paths[dataset_type]['mask'].append(pathologybin_path)
                    sample_paths[dataset_type]['pathology'].append(pathology_path)

        # Save the full paths of the samples to text files
        healthy_txt = os.path.join(os.path.dirname(path), f'{dataset_type}_healthy.txt')
        mask_txt = os.path.join(os.path.dirname(path), f'{dataset_type}_mask.txt')
        pathology_txt = os.path.join(os.path.dirname(path), f'{dataset_type}_pathology.txt')
        save_sample_paths(healthy_txt, sample_paths[dataset_type]['healthy'])
        save_sample_paths(mask_txt, sample_paths[dataset_type]['mask'])
        save_sample_paths(pathology_txt, sample_paths[dataset_type]['pathology'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_config_path", type=str, default="cfgs/dataset/train/create_train.yaml")
    parser.add_argument("--save_path", type=str, default="experiment_data")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=10)
    args = parser.parse_args()

    main(args)