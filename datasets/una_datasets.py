import os
import sys
import glob
import gc
import random
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import numpy as np

import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

from .data_utils import *
from .constants import (
    n_neutral_labels_brainseg_with_extracerebral,
    label_list_segmentation_brainseg_with_extracerebral,
    processing_funcs,
)
from FluidAnomaly.utils.misc import preprocess_cfg
from FluidAnomaly.DiffEqs.pde import AdvDiffPDE

gc.collect()


class BaseGen(Dataset):
    """
    BaseGen dataset
    """ 
    def __init__(self, gen_config_file, training_, device='cpu'):
        gen_cfg_dir = os.path.dirname(gen_config_file)
        default_gen_cfg_file = os.path.join(os.path.dirname(gen_cfg_dir), 'default.yaml')
        gen_args = preprocess_cfg([default_gen_cfg_file, gen_config_file], cfg_dir = gen_cfg_dir)
        self.gen_args = gen_args 

        self.split = gen_args.split 

        self.synth_args = self.gen_args.generator
        self.shape_gen_args = gen_args.pathology_shape_generator
        self.real_image_args = gen_args.real_image_generator
        self.synth_image_args = gen_args.synth_image_generator 

        self.augmentation_steps = vars(gen_args.augmentation_steps)

        self.input_prob = vars(gen_args.modality_probs) 
        
        self.device = device
        self.training_ = training_
        self.prepare_tasks()
        self.prepare_paths()
        self.prepare_grid() 
        self.prepare_one_hot()


    def __len__(self):
        return sum([len(self.names[i]) for i in range(len(self.names))])


    def idx_to_path(self, idx):
        cnt = 0
        for i, l in enumerate(self.datasets_len):
            if idx >= cnt and idx < cnt + l:
                return self.names[i][idx - cnt]
            else:
                cnt += l


    def prepare_paths(self):
        # Collect list of available images, per dataset
        if len(self.gen_args.dataset_names) < 1:
            datasets = [] 
            g = glob.glob(os.path.join(self.gen_args.data_root, '*' + 'T1w.nii'))
            g = [x for x in g if 'synthseg' not in x]
            for i in range(len(g)):
                filename = os.path.basename(g[i])

                dataset = filename[:filename.find('.')]
                found = False
                for d in datasets:
                    if dataset == d:
                        found = True
                if found is False:
                    datasets.append(dataset)
            print('Found ' + str(len(datasets)) + ' datasets with ' + str(len(g)) + ' scans in total')
        else:
            datasets = self.gen_args.dataset_names
        print('Dataset list', datasets)
        
        names = [] 
        if self.gen_args.split_root is not None:
            for idx, split_root in enumerate(self.gen_args.split_root):
                split_file = open(os.path.join(split_root, self.split + '.txt'), 'r')
                
                split_names = []
                for subj in split_file.readlines():
                    split_names.append(subj.strip())  

                for name in split_names:
                    names.append([os.path.join(self.gen_args.data_root[idx], 'T1', name )]) 
        else:
            for i in range(len(datasets)):
                names.append(glob.glob(os.path.join(self.gen_args.data_root, datasets[i] + '.*' + 'T1w.nii')))
        
        if 'encode_anomaly' in self.tasks:
            self.pathology_list = self.gen_args.pathology_root

        self.names = names
        self.datasets = datasets
        self.datasets_num = len(datasets)
        self.datasets_len = [len(self.names[i]) for i in range(len(self.names))]
        print('Num of data', sum([len(self.names[i]) for i in range(len(self.names))]))

        self.pathology_type = None 
        

    def prepare_tasks(self):
        self.tasks = [key for (key, value) in vars(self.gen_args.task).items() if value]
        if 'bias_field' in self.tasks and 'segmentation' not in self.tasks:

            self.tasks += ['segmentation']
        if ('encode_anomaly' in self.tasks or 'pathology' in self.tasks or 'use_original_anomaly' in self.tasks ) and self.synth_args.augment_pathology: 
            self.t = torch.from_numpy(np.arange(self.shape_gen_args.max_nt) * self.shape_gen_args.dt)
            with torch.no_grad():
                self.adv_pde = AdvDiffPDE(data_spacing=[1., 1., 1.], 
                                        perf_pattern='adv', 
                                        V_type='vector_div_free', 
                                        V_dict={},
                                        BC=self.shape_gen_args.bc, 
                                        dt=self.shape_gen_args.dt, 
                                        device=self.device
                                        )
        else:
            self.t, self.adv_pde = None, None
        for task_name in self.tasks: 
            if task_name not in processing_funcs.keys(): 
                print('Warning: Function for task "%s" not found' % task_name)


    def prepare_grid(self): 
        self.size = self.synth_args.size 

        self.res_training_data = np.array([1.0, 1.0, 1.0])

        xx, yy, zz = np.meshgrid(range(self.size[0]), range(self.size[1]), range(self.size[2]), sparse=False, indexing='ij')
        self.xx = torch.tensor(xx, dtype=torch.float)
        self.yy = torch.tensor(yy, dtype=torch.float)
        self.zz = torch.tensor(zz, dtype=torch.float)
        self.c = torch.tensor((np.array(self.size) - 1) / 2, dtype=torch.float)

        return
    
    def prepare_one_hot(self): 
        # Matrix for one-hot encoding (includes a lookup-table)
        n_labels = len(label_list_segmentation_brainseg_with_extracerebral)
        label_list_segmentation = label_list_segmentation_brainseg_with_extracerebral

        self.lut = torch.zeros(10000, dtype=torch.long)
        for l in range(n_labels):
            self.lut[label_list_segmentation[l]] = l
        self.onehotmatrix = torch.eye(n_labels, dtype=torch.float)
        
        # useless for left_hemis_only
        nlat = int((n_labels - n_neutral_labels_brainseg_with_extracerebral) / 2.0)
        self.vflip = np.concatenate([np.array(range(n_neutral_labels_brainseg_with_extracerebral)),
                                np.array(range(n_neutral_labels_brainseg_with_extracerebral + nlat, n_labels)),
                                np.array(range(n_neutral_labels_brainseg_with_extracerebral, n_neutral_labels_brainseg_with_extracerebral + nlat))])
        return

    def random_nonlinear_transform(self, photo_mode, spac):
        nonlin_scale = self.synth_args.nonlin_scale_min + np.random.rand(1) * (self.synth_args.nonlin_scale_max - self.synth_args.nonlin_scale_min)
        size_F_small = np.round(nonlin_scale * np.array(self.size)).astype(int).tolist()
        if photo_mode:
            size_F_small[1] = np.round(self.size[1]/spac).astype(int)
        nonlin_std = self.synth_args.nonlin_std_max * np.random.rand()
        Fsmall = nonlin_std * torch.randn([*size_F_small, 3], dtype=torch.float)
        F = myzoom_torch(Fsmall, np.array(self.size) / size_F_small)
        if photo_mode:
            F[:, :, :, 1] = 0

        if 'surface' in self.tasks: 
            steplength = 1.0 / (2.0 ** self.synth_args.n_steps_svf_integration)
            Fsvf = F * steplength
            for _ in range(self.synth_args.n_steps_svf_integration):
                Fsvf += fast_3D_interp_torch(Fsvf, self.xx + Fsvf[:, :, :, 0], self.yy + Fsvf[:, :, :, 1], self.zz + Fsvf[:, :, :, 2], 'linear')
            Fsvf_neg = -F * steplength
            for _ in range(self.synth_args.n_steps_svf_integration):
                Fsvf_neg += fast_3D_interp_torch(Fsvf_neg, self.xx + Fsvf_neg[:, :, :, 0], self.yy + Fsvf_neg[:, :, :, 1], self.zz + Fsvf_neg[:, :, :, 2], 'linear') 
            F = Fsvf
            Fneg = Fsvf_neg
        else:
            Fneg = None
        return F, Fneg
    
    def generate_deformation(self, setups, shp, aff, mniaffine, flip2orig, training_):
        # generate nonlinear deformation 
        if self.synth_args.nonlinear_transform and training_:
            F, Fneg = self.random_nonlinear_transform(setups['photo_mode'], setups['spac']) 
        else:
            F, Fneg = None, None

        # deform the image grid 
        xx2, yy2, zz2, x1, y1, z1, x2, y2, z2 = self.deform_grid(shp, F, mniaffine) 

        aff = torch.tensor(aff, dtype=torch.float)
        return {
                'A': mniaffine, 
                'F': F, 
                'Fneg': Fneg, 
                'grid': [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2],
                'aff_orig': aff,
                }

    
    def deform_grid(self, shp, F, affine): 
        # F = None
        if F is not None:
            xx1 = self.xx + F[:, :, :, 0]
            yy1 = self.yy + F[:, :, :, 1]
            zz1 = self.zz + F[:, :, :, 2]
        else:
            xx1 = self.xx
            yy1 = self.yy
            zz1 = self.zz


        xx2 = affine[0, 0] * xx1 + affine[0, 1] * yy1 + affine[0, 2] * zz1 + affine[0, 3]
        yy2 = affine[1, 0] * xx1 + affine[1, 1] * yy1 + affine[1, 2] * zz1 + affine[1, 3]
        zz2 = affine[2, 0] * xx1 + affine[2, 1] * yy1 + affine[2, 2] * zz1 + affine[2, 3]

        xx2[xx2 < 0] = 0
        yy2[yy2 < 0] = 0
        zz2[zz2 < 0] = 0
        xx2[xx2 > (shp[0] - 1)] = shp[0] - 1
        yy2[yy2 > (shp[1] - 1)] = shp[1] - 1
        zz2[zz2 > (shp[2] - 1)] = shp[2] - 1

        # Get the margins for reading images
        x1 = torch.floor(torch.min(xx2))
        y1 = torch.floor(torch.min(yy2))
        z1 = torch.floor(torch.min(zz2))
        x2 = 1+torch.ceil(torch.max(xx2))
        y2 = 1 + torch.ceil(torch.max(yy2))
        z2 = 1 + torch.ceil(torch.max(zz2))
        xx2 -= x1
        yy2 -= y1
        zz2 -= z1

        x1 = x1.cpu().numpy().astype(int)
        y1 = y1.cpu().numpy().astype(int)
        z1 = z1.cpu().numpy().astype(int)
        x2 = x2.cpu().numpy().astype(int)
        y2 = y2.cpu().numpy().astype(int)
        z2 = z2.cpu().numpy().astype(int)

        return xx2, yy2, zz2, x1, y1, z1, x2, y2, z2


    def augment_sample(self, name, I_ori, setups, deform_dict, res, target, pathol_direction = None, input_mode = 'synth', seed=None):
        # I_def is the original image
        sample = {}
        [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']
        if not isinstance(I_ori, torch.Tensor): #real image mode
            I = torch.squeeze(torch.tensor(I_ori.get_fdata()[x1:x2, y1:y2, z1:z2].astype(float), dtype=torch.float)) 

            # Deform grid
            I_def = fast_3D_interp_torch(I, xx2, yy2, zz2, 'linear')

            
            if self.pathology_type is None and 'pathology' in target and not torch.all(target['pathology'] == 0):
                target['pathology'][0][I < 1e-3] = 0
                target['pathology_prob'][0][I < 1e-3] = 0   

                I_path = self.encode_pathology(I, target['pathology'], target['pathology_prob'], pathol_direction, seed=seed) 
                I_path[I_path < 0.] = 0. 
            
                # check for nans
                if torch.isnan(I_path).any():
                    print('NANs in I_path - fixing')
                    I_def_path = I_def.clone()
                    target['pathology'] = torch.zeros_like(target['pathology'])
                    target['pathology_prob'] = torch.zeros_like(target['pathology_prob'])
                    target['pathology_file'] = 'no_pathology'
                else:
                    I_def_path = fast_3D_interp_torch(I_path, xx2, yy2, zz2, 'linear')

            else:
                I_def_path = I_def.clone()
        else: #synth image mode
            I_def = fast_3D_interp_torch(I, xx2, yy2, zz2)
            I_def_path = fast_3D_interp_torch(I, xx2, yy2, zz2)
        
        I_def[I_def < 0.] = 0.
        I_final = I_def / torch.max(I_def)
        I_def_path[I_def_path < 0.] = 0.
        I_final_path = I_def_path / torch.max(I_def_path)

        sample.update({'input_healthy': I_final[None], 
                       'input_pathol': I_final_path[None]})

        return sample, target
    
    
    def get_setup_params(self): 

        hemis = 'both' 

        photo_mode = np.random.rand() < self.synth_args.photo_prob
            
        pathol_mode = np.random.rand() < self.synth_args.pathology_prob
        pathol_random_shape = np.random.rand() < self.synth_args.random_shape_prob
        spac = 2.5 + 10 * np.random.rand() if photo_mode else None  
        
        if photo_mode: 
            resolution = np.array([self.res_training_data[0], spac, self.res_training_data[2]])
            thickness = np.array([self.res_training_data[0], 0.1, self.res_training_data[2]])
        else:
            resolution, thickness = resolution_sampler()
        return {'resolution': resolution, 'thickness': thickness, 
                'photo_mode': photo_mode, 'pathol_mode': pathol_mode, 
                'pathol_random_shape': pathol_random_shape,
                'spac': spac, 'hemis': hemis}
    
    
    def encode_pathology(self, I, P, Pprob, pathol_direction = None, seed = None):
        if seed is None:
            seed = time.time()
        torch.manual_seed(seed)
        # check if any of I, P, Pprob contain nans
        if torch.isnan(I).any():
            print('NANs in I')
            print(torch.isnan(I).sum())
            print(I.shape)
        if torch.isnan(P).any():
            print('NANs in P')
            print(torch.isnan(P).sum())
            print(P.shape)
        if torch.isnan(Pprob).any():
            print('NANs in Pprob')
            print(torch.isnan(Pprob).sum())
            print(Pprob.shape)
        
        if pathol_direction is None: # True: T2/FLAIR-resembled, False: T1-resembled
            pathol_direction = random.choice([True, False])

        P, Pprob = torch.squeeze(P), torch.squeeze(Pprob)

        if torch.isnan(P).any():
            print('NANs in P')
            print(torch.isnan(P).sum())
            print(P.shape)
        if torch.isnan(Pprob).any():
            print('NANs in Pprob')
            print(torch.isnan(Pprob).sum())
            print(Pprob.shape)
        I_mu = (I * P).sum() / P.sum() 
        p_mask = torch.round(P).long()
        
        pth_mus = 3*I_mu/4 + I_mu/4 * torch.rand(10000, dtype=torch.float) # enforce the pathology pattern harder!
        pth_mus = pth_mus if pathol_direction else -pth_mus 
        pth_sigmas = I_mu/4 * torch.rand(10000, dtype=torch.float) 

        
        I += Pprob * (pth_mus[p_mask] + pth_sigmas[p_mask] * torch.randn(p_mask.shape, dtype=torch.float))

        I[I < 0] = 0

        return I
    
    def get_info(self, t1):
  
        t1_name = os.path.basename(t1)

        t1_name = t1_name.split('.nii')[0]
        t1_path = t1.split('T1')[0]
        
        mniaffine = os.path.join(t1_path, 'T1_affine', t1_name + '.affine.npy') 

        generation_labels = t1.split('.nii')[0] + 'generation_labels.nii' 
        segmentation_labels = t1.split('.nii')[0] + self.gen_args.segment_prefix + '.nii'
        lp_dist_map = t1.split('.nii')[0] + 'lp_dist_map.nii'
        rp_dist_map = t1.split('.nii')[0] + 'rp_dist_map.nii'
        lw_dist_map = t1.split('.nii')[0] + 'lw_dist_map.nii'
        rw_dist_map = t1.split('.nii')[0] + 'rw_dist_map.nii'
        mni_reg_x = t1.split('.nii')[0] + 'mni_reg.x.nii'
        mni_reg_y = t1.split('.nii')[0] + 'mni_reg.y.nii'
        mni_reg_z = t1.split('.nii')[0] + 'mni_reg.z.nii'

        self.modalities = {'T1': t1, 'Gen': generation_labels, 'segmentation': segmentation_labels,   
                           'distance': [lp_dist_map, lw_dist_map, rp_dist_map, rw_dist_map],
                           'registration': [mni_reg_x, mni_reg_y, mni_reg_z], "mniaffine": mniaffine}
    

        return self.modalities


    def read_input(self, idx):
        """
        determine input type according to prob (in generator/constants.py)
        Logic: if np.random.rand() < real_image_prob and is real_image_exist --> input real images; otherwise, synthesize images. 
        """
        t1_path = self.idx_to_path(idx)

        basename = os.path.basename(t1_path).split('.nii')[0]
        parent_dir = os.path.basename(os.path.dirname(os.path.dirname(t1_path)))      

        if 'use_original_anomaly' in self.tasks:
            case_name = f"{basename}"
            t1_path_gen = t1_path.split('T1')[0]
            pathology_path = os.path.join(t1_path_gen, 'pathology_probability')
            # use paired anomaly
            pathology_files = [os.path.join(pathology_path, case_name)]
            pathology_files_all = pathology_files

        elif 'encode_anomaly' in self.tasks:
            case_name = f"{parent_dir}-{basename}"

            pathology_files_all = []
            for pathology_path in self.pathology_list:              
                pathology_files = os.listdir(pathology_path)
                pathology_files = [file for file in pathology_files if file.endswith('.nii') or file.endswith('.nii.gz')]
                pathology_files = [os.path.join(pathology_path, file) for file in pathology_files]
                pathology_files = [file for file in pathology_files if '_prob' in file]

                pathology_files_all.extend(pathology_files)

        self.modalities = self.get_info(t1_path)

        input_mode = 'T1'

        img, aff, res = read_image(self.modalities['T1']) 

        mniaffine = read_affine(self.modalities['mniaffine'])

        if 'flip2orig' in self.modalities:
            flip2orig = np.loadtxt(os.path.join(self.modalities['flip2orig'], idx.split('.nii')[0] + '.txt'), delimiter=' ')
            flip2orig = torch.from_numpy(flip2orig).float() 
        else:
            flip2orig = None

        return case_name, input_mode, img, aff, res, flip2orig, mniaffine, pathology_files_all
    

    def read_and_deform_target(self, idx, target, task_name, input_mode, setups, deform_dict, linear_weights=None, training_=True):
        if not training_:
            random.seed(42)
            seed = idx
        else:
            seed = None
        exist_keys = target.keys()
        current_target = {}
        p_prob_path, augment, thres = None, False, 0.1
        if task_name == 'encode_anomaly':
            if self.pathology_type is None:
                if setups['pathol_mode']:
                    self.pathol_mode = 'synth'
                    if setups['pathol_random_shape']:
                        p_prob_path = 'random_shape'
                        current_target['p_prob_path'] = p_prob_path
                        augment, thres = False, self.shape_gen_args.pathol_thres 
                    else:
                        if len(target['pathology_prob_paths']) == 0:
                            print(f'Warning: No pathology_prob_paths found for {target["name"]}')
                            current_target['p_prob_path'] = p_prob_path
                        else:
                            filter_small_pathol = True 
                            if not filter_small_pathol:
                                p_prob_path = random.choice(target['pathology_prob_paths']) 

                            else:
                                max_retry = 20
                                for _ in range(max_retry):
                                    p_prob_path = random.choice(target['pathology_prob_paths'])
                                    img = nib.load(p_prob_path).get_fdata()
                                    mean_val = np.mean(img)
                                    if mean_val < 0.005:
                                        # print(f"Skip {p_prob_path}, mean={mean_val:.6f}")
                                        continue
                                    else:
                                        break
                                if mean_val < 0.005:
                                    print(f"Warning: All candidates too low, using last {p_prob_path}")


                            augment, thres = self.synth_args.augment_pathology, self.shape_gen_args.pathol_thres 
            else:
                print('Warning: Pathology is not encoded for non-healthy cases')
                self.pathol_mode = 'real'
                p_prob_path = os.path.join(self.modalities['pathology_prob'], self.names[idx])
                augment, thres = False, 1e-7 

            current_target = processing_funcs[task_name](exist_keys, task_name, p_prob_path, setups, deform_dict, self.device,
                                                         target = target, 
                                                         pde_augment = augment, 
                                                         pde_func = self.adv_pde, 
                                                         t = self.t, 
                                                         shape_gen_args = self.shape_gen_args, 
                                                         thres = thres,
                                                         save_orig_for_visualize = self.gen_args.save_orig_for_visualize,
                                                         seed = seed)
    
        elif task_name != 'pathology':
            if task_name in self.modalities:
                current_target = processing_funcs[task_name](exist_keys, task_name, 
                                                            self.modalities[task_name],
                                                            setups, deform_dict, self.device, 
                                                            cfg = self.gen_args, 
                                                            onehotmatrix = self.onehotmatrix, lut = self.lut, vflip = self.vflip, seed = seed)
            else:
                current_target = {task_name: 0.}
        return current_target
    
    def deform_flip2orig(self, I_flip, deform_dict):
        # deform flip to non-flip space
        M = torch.linalg.inv(deform_dict['aff_orig']) @ deform_dict['flip2orig'] @ deform_dict['aff_orig']
        xx2 = M[0,0] * self.xx + M[0,1] * self.yy + M[0,2] * self.zz + M[0,3]
        yy2 = M[1,0] * self.xx + M[1,1] * self.yy + M[1,2] * self.zz + M[1,3]
        zz2 = M[2,0] * self.xx + M[2,1] * self.yy + M[2,2] * self.zz + M[2,3] 
        I_flip2orig = fast_3D_interp_torch(I_flip, xx2, yy2, zz2, 'linear')
        return I_flip2orig
           
    def update_gen_args(self, new_args):
        for key, value in vars(new_args).items():
            vars(self.gen_args.generator)[key] = value 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  
        try:
            # read input: real or synthesized image, according to customized prob
            case_name, input_mode, img, aff, res, flip2orig, mniaffine, pathology_paths = self.read_input(idx)

            # generate random values
            setups = self.get_setup_params() #looks like these are just the synth args
            
            # sample random deformation
            deform_dict = self.generate_deformation(setups, img.shape, aff, mniaffine, flip2orig, self.training_) 

            if self.training_:
                seed = None
            else:
                seed = idx
            # read and deform target according to the assigned tasks
            target = defaultdict(lambda: None)
            target['name'] = case_name
            target['pathology_prob_paths'] = pathology_paths
            target.update(self.read_and_deform_target(idx, target, 'T1', input_mode, setups, deform_dict))
            target.update(self.read_and_deform_target(idx, target, 'T2', input_mode, setups, deform_dict)) 
            target.update(self.read_and_deform_target(idx, target, 'FLAIR', input_mode, setups, deform_dict))
            for task_name in self.tasks:
                if task_name in processing_funcs.keys() and task_name not in ['T1', 'T2', 'FLAIR']: 
                    target.update(self.read_and_deform_target(idx, target, task_name, input_mode, setups, deform_dict))
            

            # process input sample
            self.update_gen_args(self.real_image_args) # milder noise injection for real images
            sample = self.augment_sample(case_name, img, setups, deform_dict, res, target,  
                                        pathol_direction = False, input_mode = input_mode, seed=seed)

            if setups['flip'] and isinstance(target['pathology'], torch.Tensor): # flipping should happen after P has been encoded
                target['pathology'], target['pathology_prob'] = torch.flip(target['pathology'], [1]), torch.flip(target['pathology_prob'], [1]) 

            # drop pathology_prob_paths from target
            target.pop('pathology_prob_paths', None)
            target.pop('T1_shape', None)
            # target.pop('T1_affine', None)
            return self.datasets_num, input_mode, target, sample
        except Exception as e:
            print('Error in generating idx: ', idx)
            print(e)
            return None



class UNAGen(BaseGen):

    def __init__(self, gen_config_file, training_, device='cpu'):  
        super(UNAGen, self).__init__(gen_config_file, training_, device)

        self.all_samples = self.gen_args.generator.all_samples 
        self.mild_samples = self.gen_args.generator.mild_samples 
        self.mild_generator_args = self.gen_args.mild_generator
        self.severe_generator_args = self.gen_args.severe_generator
    
    def __getitem__(self, idx):
        # read input: real or synthesized image, according to customized prob 
        case_name, input_mode, img, aff, res, flip2orig, mniaffine, pathology_prob_paths = self.read_input(idx)
        
        # generate random values
        setups = self.get_setup_params()
        # sample random deformation
        deform_dict = self.generate_deformation(setups, img.shape, aff, mniaffine, flip2orig, self.training_) 

        # read and deform target according to the assigned tasks
        target = defaultdict(lambda: 1.)
        target['name'] = case_name
        target['pathology_prob_paths'] = pathology_prob_paths

        target.update(self.read_and_deform_target(idx, target, 'T1', input_mode, setups, deform_dict, self.training_))
        if 'encode_anomaly' in self.tasks:
            target.update(self.read_and_deform_target(idx, target, 'encode_anomaly', input_mode, setups, deform_dict, self.training_)) #
        elif 'use_original_anomaly' in self.tasks:
            target.update(self.read_and_deform_target(idx, target, 'encode_anomaly', input_mode, setups, deform_dict, self.training_)) #

        self.training_ = True
        if self.training_:
            seed = None
        else:
            seed = idx
        # process or generate intra-subject input samples 
        samples = []
        for i_sample in range(self.all_samples): 
            if i_sample < self.mild_samples:  
                self.update_gen_args(self.mild_generator_args)
                self.update_gen_args(self.real_image_args)
                # target.key dict_keys(['name', 'pathology_prob_paths', 'T1', 'T1_affine', 'T1_shape', 'pathology', 'pathology_prob', 'pathology_file'])
                sample, target = self.augment_sample(case_name, img, setups, deform_dict, res, target, 
                                            pathol_direction = False, input_mode = input_mode, seed = seed)
            else: 
                self.update_gen_args(self.severe_generator_args)
                self.update_gen_args(self.real_image_args)
                sample, target = self.augment_sample(case_name, img, setups, deform_dict, res, target, 
                                            pathol_direction = False, input_mode = input_mode, seed = seed)

            samples.append(sample)
        del sample
        torch.cuda.empty_cache()
        gc.collect()      
        # prepare and deform pathology
        if 'pathology' in target and isinstance(target['pathology'], torch.Tensor):
            [xx2, yy2, zz2, x1, y1, z1, x2, y2, z2] = deform_dict['grid']

            target['pathology_prob'] = fast_3D_interp_torch(target['pathology_prob'][0], xx2, yy2, zz2)[None] 
            target['pathology'] = binarize(target['pathology_prob'], thres = self.shape_gen_args.pathol_thres)

        # check that pathology_file is a key in target
        if 'pathology_file' not in target.keys() and 'encode_anomaly' in self.tasks: 
            print('pathology_file not in target for idx: ', idx) 
            target['pathology_file'] = 'None'
            samples[0]['input_pathol'] = samples[0]['input_healthy'] 
            target['pathology'] = torch.zeros_like(samples[0]['input_healthy']) 
            target['pathology_prob'] = torch.zeros_like(samples[0]['input_healthy']) 
        target = self.convert_floats_to_tensors(target, self.device)
        samples = self.convert_floats_to_tensors(samples, self.device)

        # check that samples or target are not None
        assert samples is not None, f'Samples is None for idx: {idx}'
        assert target is not None, f'Target is None for idx: {idx}'
        # check if any of the returns are None
        if any(v is None for v in [self.datasets_num, input_mode, target, samples[0]]):
            print("Warning: None in return for idx: ", idx)
            # check which one is None
            for k, v in locals().items():
                if v is None:
                    print(k)

        # add samples[0] to target
        target.update(samples[0])
        # drop pathology_prob_paths from target
        target.pop('pathology_prob_paths', None)
        target.pop('T1_shape', None)
        # target.pop('T1_affine', None)
        del samples
        torch.cuda.empty_cache()
        gc.collect()
        return target
            

    @staticmethod
    def convert_floats_to_tensors(data, device):
        """
        Recursively converts all float values in a nested structure (dict, list, tuple) to torch tensors.

        Args:
            data: The input structure (dict, list, tuple, or any value).

        Returns:
            The same structure with all float values converted to torch tensors.
        """

        if isinstance(data, float):
            return torch.tensor(data, dtype=torch.float, device=device)
        elif isinstance(data, dict):
            return {key: UNAGen.convert_floats_to_tensors(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [UNAGen.convert_floats_to_tensors(item, device) for item in data]
        elif isinstance(data, tuple):
            return tuple(UNAGen.convert_floats_to_tensors(item, device) for item in data)
        else:
            return data