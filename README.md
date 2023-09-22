# Exploring Transfer Learning in Medical Image Segmentation using Vision-Language Models

by
Kanchan Poudel*,
Manish Dhakal*,
Prasiddha Bhandari*,
Rabin Adhikari*,
Safal Thapaliya*,
Bishesh Khanal
>*Equal contribution

This repository contains the data and source code used to reproduce the results in the paper [Exploring Transfer Learning in Medical Image Segmentation using Vision-Language Models
](https://arxiv.org/abs/2308.07706).

## Abstract

Medical image segmentation with deep learning is an important and widely studied topic because segmentation enables quantifying target structure size and shape that can help in disease diagnosis, prognosis, surgery planning, and understanding.
Recent advances in the foundation Vision-Language Models (VLMs) and their adaptation to segmentation tasks in natural images with Vision-Language Segmentation Models (VLSMs) have opened up a unique opportunity to build potentially powerful segmentation models for medical images that enable providing helpful information via language prompt as input, leverage the extensive range of other medical imaging datasets by pooled dataset training, adapt to new classes, and be robust against out-of-distribution data with human-in-the-loop prompting during inference.
Although transfer learning from natural to medical images for image-only segmentation models has been studied, no studies have analyzed how the joint representation of vision-language transfers to medical images in segmentation problems and understand gaps in leveraging their full potential.

We present the first benchmark study on transfer learning of VLSMs to 2D medical images with thoughtfully collected $11$ existing 2D medical image datasets of diverse modalities with carefully presented $9$ types of language prompts from $14$ attributes.
Our results indicate that VLSMs trained in natural image-text pairs transfer reasonably to the medical domain in zero-shot settings when prompted appropriately for non-radiology photographic modalities; when finetuned, they obtain comparable performance to conventional architectures, even in X-rays and ultrasound modalities.
However, the additional benefit of language prompts during finetuning may be limited, with image features playing a more dominant role; they can better handle training on pooled datasets combining diverse modalities and are potentially more robust to domain shift than the conventional segmentation models.


## Table of contents
- [Installation](#installation)
- [Usage](#usage)
	- [Dataset prepration](#dataset-preparation)
	- [CRIS](#running-cris)
	- [CLIPSeg](#running-clipseg)
- [License](#license)
- [Citation](#citation)

## Installation

To get started, it's recommended to create a Python environment using either Conda or venv. This helps to manage dependencies and ensure a consistent runtime environment.

1. **Conda:**
  ```bash
    conda create --name your_env_name python=3.10
    conda activate your_env_name
  ```
**OR**

2. **venv:**
  ```bash
    python -m venv your_env_name
    source your_env_name/bin/activate
  ```

Once your environment is active, install the required packages from `requirements.txt` using pip:
```bash
pip install -r requirements.txt
```

# Usage

### Dataset Preparation
Before running any experiments, you must ensure the provided dataset is correctly placed in the folders.
See the [prepare_datasets.md](prepare_datasets.md) file for more details.

### **Running CRIS**

- Download pretrained models 
	- Download the CLIP model from [here](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt).
	- Download the pretrained CRIS model from [here](https://github.com/DerrickWang005/CRIS.pytorch/issues/3#issuecomment-1290130778).
	  As the authors have not released the official CRIS model, this is an unofficial pretrained model trained on the Refcoco dataset, using the standard training procedure described in the CRIS paper.

   Place both of these models inside the `pretrained` directory.
  

- The config files for all the datasets are present inside the `CRIS.pytorch/configs` folder. 

	Example config file.
	```yaml
	DATA:
	  dataset: <dataset_name>
	  train_lmdb: <path_to_train_lmdb>
	  train_split: train
	  val_lmdb: <path_to_val_lmdb>
	  val_split: val
	  mask_root: <path_to_mask>
	TRAIN:
	  # Base Arch
	  clip_pretrain: pretrain/RN50.pt
	  input_size: 416
	  word_len: 77
	  word_dim: 1024
	  vis_dim: 512
	  fpn_in: [512, 1024, 1024]
	  fpn_out: [256, 512, 1024]
	  sync_bn: True
	  # Decoder
	  num_layers: 3
	  num_head: 8
	  dim_ffn: 2048
	  dropout: 0.2
	  intermediate: False
	  # Training Setting
	  workers: 16 # data loader workers
	  workers_val: 16
	  epochs: 100
	  milestones: [100]
	  start_epoch: 0
	  batch_size: 16  # batch size for training
	  batch_size_val: 16 # batch size for validation during training, memory and speed tradeoff
	  base_lr: 0.0001
	  lr_decay: 0.1
	  lr_multi: 0.1
	  weight_decay: 0.01
	  max_norm: 0.
	  manual_seed: 0
	  print_freq: 1
	  prompt_type: p0 # this is the prompt name to use from the json file
	  log_model: False
	  resize: False 
	  # Resume & Save
	  exp_name: CRIS_R50 
	  output_folder: <output_folder> # The models are saved in this folder
	  save_freq: 1
	  weight:  # path to initial weight (default: none)
	  resume: <path_to_trained_cris_model> # path to latest checkpoint (default: none)
	  resume_optimizer: True #restores optimizer state while loading checkpoint
	  resume_scheduler: True #restores scheduler state while loading checkpoint
	  evaluate: True  # evaluate on the validation set, extra gpu memory needed and small batch_size_val is recommend
	  train_clip: True # whether to train the clip encoder
	Distributed:
	  dist_url: tcp://localhost:6745
	  dist_backend: 'nccl'
	  multiprocessing_distributed: True
	  world_size: 1
	  rank: 0
	TEST:
	  test_split: <test_split_name>
	  test_lmdb: <path_to_test_lmdb>
	  visualize: True
	```
- Create/change the config files for individual datasets.

- Edit path arguments and prompt type in config

  - Data: Paths to fill are commented
  
  - Train: Change the following args 
	- output_folder
	- prompt_type
- In the `CRIS.pytorch/utils/dataset.py` file, add the dataset information inside the `info` dictionary as:
  ```json
	"dataset_name": {"train": 1330, "val": 420, "testA": 427, "testB": 427}
  ```
  The values of `train`, `val`, and `testA/testB` in this dictionary is the number of images in the respective splits.
  The `dataset_name` key is the same as `DATA.dataset_name` in the config file.
  
  **NOTE: Without adding this split information inside the dataset.py, the training/testing code does not work.**
  
  
- Run scripts
    - For inference
    	Use the script `./CRIS.pytorch/test.sh` as a sample and edit the necessary paths.

  	This script uses the `test.py` file for inference.
  	For zero-shot inference, use `TRAIN.resume: pretrain/cris_best.pth` in the config file.
  	This is the downloaded pretrained CRIS model.
    - For fine-tuning
    	Use the script `./CRIS.pytorch/finetune.sh` as a sample and edit the necessary paths.

### **Running CLIPSeg**

CLIPSeg uses the pretrained models [available on HuggingFace](https://huggingface.co/docs/transformers/model_doc/clipseg).
No need to download the models 🥳.

- Prepare configs for the datasets.

  The configs for the datasets are inside the `VL-Seg/configs/datamodule` folder.
  Each config file is named as `img_mask_dataset-name.yaml`.

  Example config file:
```yaml
_target_: src.datamodules.BaseDataModule
train_dataset:
  _target_: src.datamodules.datasets.ImageMaskDataset
  images_dir: <img_dir_path>
  masks_dir: <mask_dir_path>
  caps_file: <train_json_file>
  img_size: ${img_size}
  transforms: ${train_img_transforms}
  

val_dataset:
  _target_: src.datamodules.datasets.ImageMaskDataset
  images_dir: <img_dir_path>
  masks_dir: <mask_dir_path>
  caps_file: <val_json_file>
  img_size: ${img_size}
  transforms: ${val_img_transforms}

test_dataset:
  _target_: src.datamodules.datasets.ImageMaskDataset
  images_dir: <img_dir_path>
  masks_dir: <mask_dir_path>
  caps_file: <test_json_file>
  img_size: ${img_size}
  transforms: ${test_img_transforms}

batch_size: 16
train_val_split: [0.8, 0.2]
num_workers: 4
pin_memory: True
```

- Add the `dataset_name` and `prompt_types` in the `VL-Seg/scripts/configs.sh` file after adding the config file for a particular dataset.

- Zero Shot Segmentation

	To perform zero-shot segmentation, you can use the provided script. 
	
	Navigate to the `scripts` directory in the `VL-Seg` folder.
	  ```bash
	  cd VL-Seg/scripts
	  ```
	
	Open a terminal and navigate to the project directory, then execute the following command:
	```bash
	bash zss.sh
	```
	This script will initiate the zero-shot segmentation process and produce the desired results.

- Fine-Tuning

	If you need to run fine-tuning for your model, you can do so using the following script:
	```bash
	bash finetune.sh
	```
	This script will start the fine-tuning process, essential for customizing the model for specific tasks.


## License
All Python source code (including `.py` and `.ipynb` files) is made available
under the MIT license.
You can freely use and modify the code, without warranty, so long as you provide attribution to the authors.
See [LICENSE](LICENSE) for the full license text.

The manuscript text (including all LaTeX files), figures, and data/models produced as part of this research are available under the [Creative Commons Attribution 4.0 License (CC-BY)](https://creativecommons.org/licenses/by/4.0/).
See [LICENSE](datasets/anns/LICENSE) for the full license text.
## Citation
```
@article{poudel2023exploring,
  title={Exploring Transfer Learning in Medical Image Segmentation using Vision-Language Models},
  author={Poudel, Kanchan and Dhakal, Manish and Bhandari, Prasiddha and Adhikari, Rabin and Thapaliya, Safal and Khanal, Bishesh},
  journal={arXiv preprint arXiv:2308.07706},
  year={2023}
}
```
