# save new affine matrices of skull stripped images 
import os
import torch
import argparse
import numpy as np
import nibabel as nib

from os.path import join
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def getM(ref, mov):

    zmat = np.zeros(ref.shape[::-1])
    zcol = np.zeros([ref.shape[1], 1])
    ocol = np.ones([ref.shape[1], 1])
    zero = np.zeros(zmat.shape)


    A = np.concatenate([
        np.concatenate([np.transpose(ref), zero, zero, ocol, zcol, zcol], axis=1),
        np.concatenate([zero, np.transpose(ref), zero, zcol, ocol, zcol], axis=1),
        np.concatenate([zero, zero, np.transpose(ref), zcol, zcol, ocol], axis=1)], axis=0)

    b = np.concatenate([np.transpose(mov[0, :]), np.transpose(mov[1, :]), np.transpose(mov[2, :])], axis=0)

    x = np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.matmul(np.transpose(A), b))

    M = np.stack([
        [x[0], x[1], x[2], x[9]],
        [x[3], x[4], x[5], x[10]],
        [x[6], x[7], x[8], x[11]],
        [0, 0, 0, 1]])

    return M

def load_volume(path_volume, im_only=True, squeeze=True, dtype=None, aff_ref=None):

    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

    if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(path_volume)
        if squeeze:
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
    else:  # npz
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        if 'int' in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)

    if im_only:
        return volume
    else:
        return volume, aff, header
    
def easy_reg(synseg_file, input_file, output_file):
    # path labels
    atlas_volsize = [160, 160, 192]
    atlas_aff = np.matrix([[-1, 0, 0, 79], [0, 0, 1, -104], [0, -1, 0, 79], [0, 0, 0, 1]])
    labels = np.array([2,4,5,7,8,10,11,12,13,14,15,16,17,18,26,28,41,43,44,46,47,49,50,51,52,53,54,58,60,
                                        1001,1002,1003,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,
                                        2001,2002,2003,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035])
    nlab = len(labels)
    atlasCOG = np.array([[-28.,-18.,-37.,-19.,-27.,-19.,-23.,-31.,-26.,-2.,-3.,-3.,-29.,-26.,-14.,-14.,24.,14.,31.,12.,18.,14.,19.,26.,21.,25.,22.,11.,8.,-52.,-6.,-36.,-7.,-24.,-37.,-39.,-52.,-9.,-27.,-26.,-14.,-8.,-59.,-28.,-7.,-49.,-43.,-47.,-12.,-46.,-6.,-43.,-10.,-7.,-33.,-11.,-23.,-55.,-50.,-10.,-29.,-46.,-38.,48.,4.,31.,3.,21.,33.,37.,47.,3.,24.,20.,8.,4.,54.,21.,5.,45.,38.,46.,8.,45.,3.,38.,6.,4.,29.,9.,19.,51.,49.,10.,24.,43.,33.],
                        [-30.,-17.,-13.,-36.,-40.,-22.,-3.,-5.,-9.,-14.,-31.,-21.,-15.,-1.,3.,-16.,-32.,-20.,-14.,-37.,-42.,-24.,-3.,-6.,-10.,-15.,-2.,3.,-17.,-44.,-5.,-15.,-71.,2.,-29.,-70.,-23.,-44.,-73.,22.,-57.,27.,-19.,-23.,-45.,4.,31.,20.,-68.,-38.,-33.,-26.,-60.,23.,22.,0.,-72.,-12.,-49.,49.,17.,-25.,-3.,-42.,-1.,-16.,-76.,0.,-34.,-69.,-16.,-44.,-73.,22.,-56.,28.,-18.,-25.,-45.,-3.,30.,14.,-69.,-37.,-32.,-30.,-60.,21.,21.,0.,-72.,-11.,-49.,48.,15.,-27.,-3.],
                        [12.,14.,-13.,-41.,-51.,1.,13.,3.,1.,0.,-40.,-28.,-15.,-10.,2.,-7.,11.,14.,-12.,-40.,-51.,2.,14.,4.,2.,-14.,-10.,4.,-7.,-8.,32.,40.,-14.,-21.,-28.,-4.,-28.,-3.,-35.,3.,-29.,4.,-17.,-21.,35.,18.,9.,20.,-24.,28.,25.,34.,7.,18.,35.,48.,16.,-5.,12.,22.,-18.,1.,4.,-12.,32.,43.,-11.,-21.,-29.,-3.,-27.,0.,-34.,3.,-25.,6.,-18.,-20.,36.,18.,11.,20.,-20.,26.,25.,34.,4.,24.,34.,47.,17.,-5.,10.,20.,-18.,0.,4.]])

    synseg_buffer, synseg_aff, synseg_h = load_volume(synseg_file, im_only=False, squeeze=True, dtype=None, aff_ref=None)

    synsegCOG = np.zeros([4, nlab])
    ok = np.ones(nlab)
    for l in range(nlab):
        aux = np.where(synseg_buffer == labels[l])
        if len(aux[0]) > 50:
            synsegCOG[0, l] = np.median(aux[0])
            synsegCOG[1, l] = np.median(aux[1])
            synsegCOG[2, l] = np.median(aux[2])
            synsegCOG[3, l] = 1
        else:
            ok[l] = 0
    synsegCOG = np.matmul(synseg_aff, synsegCOG)[:-1, :]
    Mref = getM(atlasCOG[:, ok > 0], synsegCOG[:, ok > 0])

    II, JJ, KK = np.meshgrid(np.arange(atlas_volsize[0]), np.arange(atlas_volsize[1]), np.arange(atlas_volsize[2]), indexing='ij')
    II = torch.tensor(II, device='cpu')
    JJ = torch.tensor(JJ, device='cpu')
    KK = torch.tensor(KK, device='cpu')
    synsr, synsr_aff, synsr_h = load_volume(input_file, im_only=False, squeeze=True, dtype=None, aff_ref=None)
    synsr = torch.tensor(synsr)

    affine = torch.tensor(np.matmul(np.linalg.inv(synsr_aff), np.matmul(Mref, atlas_aff)), device='cpu')
    np.save(output_file, affine.numpy())

def process_subject(subj, data_path, input_path, new_affine_path):
    subj_name = subj.split('.')[0]
    seg_file = os.path.join(data_path, subj)
    img_file = os.path.join(input_path, subj)
    out_file = os.path.join(new_affine_path, f"{subj_name}.affine.npy")

    if os.path.exists(out_file):
        return subj_name, "skipped"

    try:
        easy_reg(seg_file, img_file, out_file)
        return subj_name, "done"
    except Exception as e:
        return subj_name, f"error: {e}"

def main():
    parser = argparse.ArgumentParser(description="Run affine processing for multiple subjects.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input T1 directory")
    parser.add_argument("--label_path", type=str, required=True,
                        help="Path to segmentation label maps")
    parser.add_argument("--new_affine_path", type=str, required=True,
                        help="Path to save affine .npy outputs")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")

    args = parser.parse_args()

    input_path = args.input_path
    label_path = args.label_path
    new_affine_path = args.new_affine_path

    os.makedirs(new_affine_path, exist_ok=True)

    subjlist = [x for x in os.listdir(label_path)
                if x.endswith(".nii.gz") or x.endswith(".nii")]
    print(f"Found {len(subjlist)} subjects")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_subject, subj, label_path, input_path, new_affine_path): subj
            for subj in subjlist
        }

        for future in as_completed(futures):
            subj, status = future.result()
            print(f"{subj}: {status}")


if __name__ == "__main__":
    main()