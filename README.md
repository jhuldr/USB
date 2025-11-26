## <p align="center">[Unified Synthetic Brain]</p>

**<p align="center">Jun Wang<sup>1</sup>, Peirong Liu<sup>1</sup></p>**

<p align="center">
<sup>1</sup>Johns Hopkins University
</p>

<p align="center">
  <img src="./assets/showcase.png" alt="drawing", width="850"/>
</p>


## Downloads
Please download USB's weights ('./checkpoints/usb_lesion.pth', './assets/checkpoints/usb_brain.pth') and testing images ('./test_samples') in this [Google Drive folder](x), then move them into the './assets' folder in this repository. We also provided original images for generating these testing samples in './data'.



## Environment
Training and evaluation environment: Python 3.11.4, PyTorch 2.0.1, CUDA 12.2. Run the following command to install required packages.
```
conda create -n USB python=3.11
conda activate USB

cd /path/to/usb
pip install -r requirements.txt
```

## Demo


### Fluid-Driven Anatomy Randomization Generator

```
cd /path/to/usb

python scripts/mni_mapping.py \
    --input_path assets/data/hcp/T1 \
    --label_path assets/data/hcp/label_maps_segmentation \
    --new_affine_path assets/data/hcp/T1_affine \
    --workers 8

python scripts/demo_create_dataset.py \
    --data_config_path cfgs/dataset/test/create_test.yaml \
    --save_path assets

```

### Generation and Editing


```
cd /path/to/usb

# unconditional generation
python scripts/demo_test.py \
    --mode uncond_gen \
    --config_path cfgs/trainor/test/demo_test.yaml

# conditional generation    
python scripts/demo_test.py \
    --mode cond_gen \
    --config_path cfgs/trainor/test/demo_test.yaml

# pathology-to-healthy editing  
python scripts/demo_test.py \
    --mode p2h_edit \
    --config_path cfgs/trainor/test/demo_test.yaml

# healthy-to-pathology editing  
python scripts/demo_test.py \
    --mode h2p_edit \
    --config_path cfgs/trainor/test/demo_test.yaml
```

## Training on Synthetic Data and/or Real Data

TODO
