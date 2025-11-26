import os
import sys

import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt

from torch.optim import lr_scheduler
from torch.utils import data
from tqdm import tqdm

from skimage import morphology
from skimage.transform import resize

from Trainor.models import LDM_3D
from datasets import USBData
from utils.get_betas import get_betas
from utils.denoise import (
    denoise_uncond,
    denoise_cond,
    denoise_p2h,
    denoise_h2p,
)

from autoencoders import AutoencoderKLAD

def load_nii_as_tensor(path, target_shape=(160,160,160)):

        nii = nib.load(path)
        img = nii.get_fdata().astype(np.float32)

        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

        # img = F.interpolate(img, size=target_shape, mode='trilinear', align_corners=False)
        return img.squeeze(0)

def get_slice(slice):

    target_size = (230, 182)
    slice = resize(slice, target_size, order=1, preserve_range=True, anti_aliasing=True)

    h, w = slice.shape
    max_side = max(h, w)
    pad_top = (max_side - h) // 2
    pad_bottom = max_side - h - pad_top
    pad_left = (max_side - w) // 2
    pad_right = max_side - w - pad_left

    slice = np.pad(slice, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    return slice


def save_volume_and_slice(
        volume,
        result_dir,
        name_prefix,
        slice_idx=60,
        do_binary=False,
        affine=None
    ):

    if affine is None:
        affine = np.eye(4)

    vol_path = os.path.join(result_dir, f"{name_prefix}.nii.gz")

    nib.save(nib.Nifti1Image(volume, affine=affine), vol_path)

    mid_slice = np.rot90(volume[:, slice_idx, :])

    if do_binary:
        mid_slice = (mid_slice - volume.min()) / (volume.max() - volume.min() + 1e-8)
        mid_slice = (mid_slice > 0.5)
        mid_slice = morphology.remove_small_holes(mid_slice, area_threshold=50).astype(np.float32)

    mid_slice = get_slice(mid_slice)

    slice_path = os.path.join(result_dir, f"{name_prefix}.png")
    

    plt.imsave(slice_path, mid_slice, cmap="gray")

    return vol_path, slice_path


class Test():
    def __init__(self, args):
        # load configs
        self.model_config = args.model
        self.diffusion_config = args.diffusion
        self.data_config = args.data

        if args.mode == "uncond_gen":
            self.sample_config = args.uncond_gen

        elif args.mode == "cond_gen":
            self.sample_config = args.cond_gen
        elif args.mode == "p2h_edit":
            self.edit_config = args.p2h_edit
        elif args.mode == "h2p_edit":
            self.edit_config = args.h2p_edit
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.betas = get_betas(self.diffusion_config).to(self.device)
        self.num_diffusion_timesteps = self.diffusion_config.num_diffusion_timesteps

        # load VAE
        if not os.path.exists(self.model_config.vae_path):
            raise ValueError(f"Model not found at {self.model_config.vae_path}")
        self.vae = AutoencoderKLAD.load_from_checkpoint(self.model_config.vae_path, cfg=self.model_config.vae_config_path, input_dim=[(160, 160, 160)])
        self.vae.eval() 
        self.vae.requires_grad_(False)
        
    def uncond_gen(self):
        """
        Unconditional Generation
        """

        model_config = self.model_config
        sample_config = self.sample_config
        device = self.device
        img_num = sample_config.img_num
        num_diffusion_timesteps = self.num_diffusion_timesteps
        betas = self.betas
        eta = sample_config.eta
        assert sample_config.skip_type == 'uniform' or sample_config.skip_type == 'quad', print(
            sample_config.skip_type)
        
        if sample_config.skip_type == 'uniform':
            skip = num_diffusion_timesteps // sample_config.num_sample_timesteps
            denoise_t_list = range(0, num_diffusion_timesteps, skip)
        elif sample_config.multi_sample.skip_type == "quad":
            denoise_t_list = (np.linspace(0, np.sqrt(num_diffusion_timesteps * 0.8),
                                          sample_config.num_sample_timesteps) ** 2)
            denoise_t_list = [int(s) for s in list(denoise_t_list)]
        else:
            raise NotImplementedError("Skip type not defined")
        
        x_model = LDM_3D(in_channels=model_config.in_channels,
                        out_channels=model_config.in_channels,
                        num_channels=model_config.num_channels,
                        attention_levels=model_config.attention_levels,
                        num_res_blocks=model_config.num_res_blocks,
                        num_head_channels=model_config.num_head_channels, 
                        with_conditioning=False,
                        learn_sigma=model_config.learn_sigma)
        x_model_path = model_config.x_model_path
        x_states = torch.load(x_model_path)
        x_model = x_model.to(device)
        x_model = torch.nn.DataParallel(x_model)
        x_model.load_state_dict(x_states[0], strict=True)


        y_model = LDM_3D(in_channels=model_config.in_channels,
                        out_channels=model_config.in_channels,
                        num_channels=model_config.num_channels,
                        attention_levels=model_config.attention_levels,
                        num_res_blocks=model_config.num_res_blocks,
                        num_head_channels=model_config.num_head_channels,
                        with_conditioning=False,
                        learn_sigma=model_config.learn_sigma)
        y_model_path = model_config.y_model_path
        y_states = torch.load(y_model_path)
        y_model = y_model.to(device)
        y_model = torch.nn.DataParallel(y_model)
        y_model.load_state_dict(y_states[0], strict=True)
        x_model.eval()
        y_model.eval()

        with torch.no_grad():
            batch_size = sample_config.batch_size
            batch_num = np.ceil(img_num / batch_size).astype(int)
            result_dir = sample_config.result_dir
            for batch in range(batch_num):
                n_batch = min(batch_size, img_num - batch * batch_size)

                xt = torch.randn((n_batch, model_config.in_channels) + tuple(model_config.initial_resolution)).to(device)
                yt = torch.randn((n_batch, model_config.in_channels) + tuple(model_config.initial_resolution)).to(device)
                xts, yts, x0s, y0s = denoise_uncond(x_model, y_model, xt, yt, denoise_t_list, betas, eta)

                x0, y0 = x0s[-1], y0s[-1]
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                self.vae = self.vae.to(device)

                x0 = 1 / 0.18215 * x0
                x0 = self.vae.decode([x0])[0]._sample()

                y0 = 1 / 0.18215 * y0
                y0 = self.vae.decode([y0])[0]._sample()
                
                x0 = x0.detach().cpu().numpy()
                y0 = y0.detach().cpu().numpy()                   
                
                for i in tqdm(range(x0.shape[0]), desc='saving'):

                    volume_x = x0[i, 0]
                    volume_y = y0[i, 0]

                    idx = batch * batch_size + i

                    save_volume_and_slice(volume_x, result_dir, f"x0_{idx}", do_binary=True)
                    save_volume_and_slice(volume_y, result_dir, f"y0_{idx}", do_binary=False)
    

    def cond_gen(self):
        """
        Conditional Generation
        """

        model_config = self.model_config
        sample_config = self.sample_config
        device = self.device
        img_num = sample_config.img_num
        num_diffusion_timesteps = self.num_diffusion_timesteps
        betas = self.betas
        eta = sample_config.eta

        assert sample_config.skip_type in ['uniform', 'quad'], print(sample_config.skip_type)
        
        if sample_config.skip_type == 'uniform':
            skip = num_diffusion_timesteps // sample_config.num_sample_timesteps
            denoise_t_list = range(0, num_diffusion_timesteps, skip)
        elif sample_config.skip_type == 'quad':
            denoise_t_list = (np.linspace(0, np.sqrt(num_diffusion_timesteps * 0.8),
                                        sample_config.num_sample_timesteps) ** 2)
            denoise_t_list = [int(s) for s in list(denoise_t_list)]
        else:
            raise NotImplementedError("Skip type not defined")

        y_model = LDM_3D(in_channels=model_config.in_channels,
                        out_channels=model_config.in_channels,
                        num_channels=model_config.num_channels,
                        attention_levels=model_config.attention_levels,
                        num_res_blocks=model_config.num_res_blocks,
                        num_head_channels=model_config.num_head_channels,
                        with_conditioning=False,
                        learn_sigma=model_config.learn_sigma)

        y_states = torch.load(model_config.y_model_path)
        y_model = torch.nn.DataParallel(y_model.to(device))
        y_model.load_state_dict(y_states[0], strict=True)
        
        y_model.eval()

        x_file_list = sample_config.x_data_path
        with open(sample_config.x_data_path, 'r') as f: 
            x_files = [line.strip() for line in f.readlines()]

        x_files = sorted(x_files)
        print(f"Found {len(x_files)} NIfTI files in {x_file_list}")

        with torch.no_grad():
            result_dir = sample_config.result_dir
            os.makedirs(result_dir, exist_ok=True)

            for idx, nii_file in enumerate(tqdm(x_files, desc="Generating")):
                x0 = load_nii_as_tensor(nii_file).to(device)
                x0 = x0.unsqueeze(0)

                self.vae = self.vae.to(device)
                x0 = self.vae.encode([x0])[0]._sample().mul_(0.18215) 
                x0 = x0.to(device)

                yt = torch.randn_like(x0).to(device)

                yts, y0s = denoise_cond(y_model, x0, yt, denoise_t_list, betas, eta)

                x0 = 1 / 0.18215 * x0
                x0 = self.vae.decode([x0])[0]._sample()

                y0 = y0s[-1]
                y0 = 1 / 0.18215 * y0
                y0 = self.vae.decode([y0])[0]._sample()

                x0 = x0.detach().cpu().numpy().astype(np.float32)
                y0 = y0.detach().cpu().numpy().astype(np.float32)

                volume_x = x0[0, 0]
                volume_y = y0[0, 0]
                
                case_name = os.path.splitext(os.path.basename(nii_file))[0]

                save_volume_and_slice(volume_x, result_dir, f"x0_{case_name}_{idx}", do_binary=True)
                save_volume_and_slice(volume_y, result_dir, f"y0_{case_name}_{idx}", do_binary=False)
                        
    def p2h_edit(self):
        """
        Pathology-to-Healthy Editing
        """
        data_config = self.data_config
        model_config = self.model_config
        edit_config = self.edit_config
        data_config.data_file = edit_config.y_data_path

        device = self.device
        num_diffusion_timesteps = self.num_diffusion_timesteps
        betas = self.betas
        eta = edit_config.eta

        result_dir = edit_config.result_dir

        assert edit_config.skip_type in ['uniform', 'quad'], f"Unknown skip_type: {edit_config.skip_type}"
        
        if edit_config.skip_type == 'uniform':
            skip = num_diffusion_timesteps // edit_config.num_sample_timesteps
        elif edit_config.skip_type == 'quad':
            skip = max(1, int(np.sqrt(num_diffusion_timesteps) // np.sqrt(edit_config.num_sample_timesteps)))

        dataset = USBData(data_config, training_=False, device=device)
        edit_dataloader = data.DataLoader(
            dataset, batch_size=edit_config.batch_size, shuffle=False, num_workers=edit_config.num_workers
        )
        
        y_model = LDM_3D(in_channels=model_config.in_channels,
                        out_channels=model_config.in_channels,
                        num_channels=model_config.num_channels,
                        attention_levels=model_config.attention_levels,
                        num_res_blocks=model_config.num_res_blocks,
                        num_head_channels=model_config.num_head_channels,
                        with_conditioning=False,
                        learn_sigma=model_config.learn_sigma)

        y_model_path = model_config.y_model_path
        y_states = torch.load(y_model_path, map_location=device)
        y_model = torch.nn.DataParallel(y_model.to(device))
        y_model.load_state_dict(y_states[0], strict=True)
        y_model.eval()


        t_start = getattr(edit_config, "t_start", 300)
        skip = num_diffusion_timesteps // edit_config.num_sample_timesteps
        denoise_t_list = range(0, t_start, skip)

        with torch.no_grad():
            with tqdm(total=len(edit_dataloader), desc=f'Editing', file=sys.__stdout__) as pbar:
                for i, batch in enumerate(edit_dataloader):

                    y_p = batch[edit_config.y].to(device)

                    self.vae = self.vae.to(device)

                    y_p = self.vae.encode([y_p])[0]._sample().mul_(0.18215).to(device)
                    
                    t = torch.full((y_p.shape[0],), t_start, device=device, dtype=torch.long)
                    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
                    e_y = torch.randn_like(y_p)
                    yt = y_p * a.sqrt() + e_y * (1.0 - a).sqrt()

                    xt = torch.zeros_like(y_p).to(device)

                    xts, yts, x0s, y0s = denoise_p2h(
                        y_model, xt, yt, y_p, denoise_t_list, betas, eta,
                        alpha0=20.0, 
                        decay=0.5,
                    )

                    y0 = y0s[-1]
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    self.vae = self.vae.to(device)

                    y0 = 1 / 0.18215 * y0
                    y0 = self.vae.decode([y0])[0]._sample()

                    y_p = 1 / 0.18215 * y_p
                    y_p = self.vae.decode([y_p])[0]._sample()
                    
                    y0 = y0.detach().cpu().numpy()
                    y_p = y_p.detach().cpu().numpy()

                    for i in tqdm(range(y0.shape[0]), desc='saving'):
                        volume_y_h = y0[i, 0]
                        volume_y_p = y_p[i, 0]

                        case_name = batch['img_name'][i] + '_' + batch['pathol_name'][i]
                        save_volume_and_slice(volume_y_h, result_dir, f"y_h_{case_name}", do_binary=False)
                        save_volume_and_slice(volume_y_p, result_dir, f"y_p_{case_name}", do_binary=False)

    def h2p_edit(self):
        """
        Healthy-to-Pathology Editing
        """
        data_config = self.data_config
        model_config = self.model_config
        edit_config = self.edit_config
        data_config.data_file = edit_config.y_data_path

        device = self.device
        num_diffusion_timesteps = self.num_diffusion_timesteps
        betas = self.betas
        eta = edit_config.eta

        result_dir = edit_config.result_dir

        assert edit_config.skip_type in ['uniform', 'quad'], f"Unknown skip_type: {edit_config.skip_type}"
        
        if edit_config.skip_type == 'uniform':
            skip = num_diffusion_timesteps // edit_config.num_sample_timesteps
        elif edit_config.skip_type == 'quad':
            skip = max(1, int(np.sqrt(num_diffusion_timesteps) // np.sqrt(edit_config.num_sample_timesteps)))

        dataset = USBData(data_config, training_=False, device=device)
        edit_dataloader = data.DataLoader(
            dataset, batch_size=edit_config.batch_size, shuffle=False, num_workers=edit_config.num_workers
        )

        y_model = LDM_3D(in_channels=model_config.in_channels,
                        out_channels=model_config.in_channels,
                        num_channels=model_config.num_channels,
                        attention_levels=model_config.attention_levels,
                        num_res_blocks=model_config.num_res_blocks,
                        num_head_channels=model_config.num_head_channels,
                        with_conditioning=False,
                        learn_sigma=model_config.learn_sigma)

        y_model_path = model_config.y_model_path
        y_states = torch.load(y_model_path, map_location=device)
        y_model = torch.nn.DataParallel(y_model.to(device))
        y_model.load_state_dict(y_states[0], strict=True)
        y_model.eval()

        t_start = getattr(edit_config, "t_start", 500)
        skip = num_diffusion_timesteps // edit_config.num_sample_timesteps
        denoise_t_list = range(0, t_start, skip)

        with torch.no_grad():
            with tqdm(total=len(edit_dataloader), desc=f'Editing', file=sys.__stdout__) as pbar:
                for i, batch in enumerate(edit_dataloader):

                    y_h = batch['img_wo_pathol'].to(device)

                    x0_ori = batch[edit_config.x].to(device)

                    self.vae = self.vae.to(device)

                    y_h = self.vae.encode([y_h])[0]._sample().mul_(0.18215).to(device)
                    x0_ori = self.vae.encode([x0_ori])[0]._sample().mul_(0.18215).to(device)

                    t = torch.full((y_h.shape[0],), t_start, device=device, dtype=torch.long)
                    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
                    e_y = torch.randn_like(y_h)
                    yt = y_h * a.sqrt() + e_y * (1.0 - a).sqrt()
                    xt = x0_ori

                    xts, yts, x0s, y0s = denoise_h2p(
                        y_model, xt, yt, y_h, denoise_t_list, betas, eta,
                        lesion_free_scale=1
                    )

                    x0, y0 = x0s[-1], y0s[-1]

                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    self.vae = self.vae.to(device)

                    x0 = 1 / 0.18215 * x0
                    x0 = self.vae.decode([x0])[0]._sample()

                    y0 = 1 / 0.18215 * y0
                    y0 = self.vae.decode([y0])[0]._sample()

                    y_h = 1 / 0.18215 * y_h
                    y_h = self.vae.decode([y_h])[0]._sample()
                    
                    x0 = x0.detach().cpu().numpy()
                    y0 = y0.detach().cpu().numpy()
                    y_h = y_h.detach().cpu().numpy()                             
                    
                    for i in tqdm(range(y0.shape[0]), desc='saving'):
                        volume_x = x0[i, 0]
                        volume_y_p = y0[i, 0]
                        volume_y_h = y_h[i, 0]

                        case_name = batch['img_name'][i] + '_' + batch['pathol_name'][i]
                        save_volume_and_slice(volume_y_h, result_dir, f"y_h_{case_name}", do_binary=False)
                        save_volume_and_slice(volume_y_p, result_dir, f"y_p_{case_name}", do_binary=False)