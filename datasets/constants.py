from .data_utils import *

augmentation_funcs = {
    'gamma': add_gamma_transform,
    'bias_field': add_bias_field,
    'resample': resample_resolution,
    'noise': add_noise,
}

processing_funcs = {
    'T1': read_and_deform_image,
    'T2': read_and_deform_image,
    'FLAIR': read_and_deform_image,
    'encode_anomaly': read_pathology, 
    'mask': read_and_deform_mask
}


# with csf # NOTE old version (FreeSurfer standard), non-vast
label_list_segmentation = [0,14,15,16,24,77,85,   2, 3, 4, 7, 8, 10,11,12,13,17,18,26,28,   41,42,43,46,47,49,50,51,52,53,54,58,60] # 33
n_neutral_labels = 7


## NEW VAST synth
label_list_segmentation_brainseg_with_extracerebral =  [0, 11, 12, 13, 16, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46,
                                   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 17, 47, 49, 51, 53, 55,
                                   18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 48, 50, 52, 54, 56] 
n_neutral_labels_brainseg_with_extracerebral = 20

label_list_segmentation_brainseg_left = [0, 1, 2, 3, 4, 7, 8, 9, 10, 14, 15, 17, 31, 34, 36, 38, 40, 42]

