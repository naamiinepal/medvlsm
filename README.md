# Exploring Transfer Learning in Medical Image Segmentation using Vision-Language Models

CRIS is a referring image segmentation model architecture. The model is originally trained on refcoco, refcoco+ and G-ref dataset. We have adapted the CRIS' official implentation from [here](https://github.com/DerrickWang005/CRIS.pytorch) to finetune and evaluate model for medical image segmentation.

## Table of contents
- [Installation](#installation)
- [Usage](#usage)
	- [Dataset prepration](#dataset-preparation)
	- [CRIS](#runnning-cris)
	- [CLIPSeg](#running-clipseg)
- [Datasets used](#datasets-used-in-our-experiments)
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
Before running any experiments, you need to ensure that the provided dataset is correctly placed within the `data` folder at the root of the project. The directory structure of the `data` folder should look like this:
```
data/
│
├── bkai_polyp/
│   ├── anns/
│   │   ├── testA.json
│   │   ├── train.json
│   │   └── val.json
│   ├── images/
│   └── masks/
│
├── [other dataset folders...]
│
└── kvasir_polyp/
    ├── anns/
    │   ├── testA.json
    │   ├── train.json
    │   └── val.json
    ├── images/
    └── masks/
```
Each dataset folder (`bkai_polyp`, `busi`, `camus`, etc.) contains three sub-directories: `anns`, `images`, and `masks`. The anns directory contains prompt files (`testA.json`, `train.json`, `val.json`), while `images` and `masks` hold input images and target masks respectively.
For more details see `./CRIS.pytorch/blob/master/tools/prepare_datasets.md`.

### **RUNNING CRIS**

1. Get this repository in your system.

2. Dataset preparation:
First of all download the required datasets using the referecnes given in the paper. Some of the datasets may require requesting permissions so please follow accordingly.
After downloading dataset, prepare dataset in Ref-COCO format.
Annotations format used in our experiments is slightly different than original annotation format.
For reference format used in our experiment refer sample at `./CRIS.pytorch/datasets/anns/kvasir_polyp_80_10_10/train.json`(also see `val.json`, `testA.json` and `testB.json`).
In the key `prompts` of each of the annotation, there should be key value pair of prompts of each type (`P0, P1, ..., Pn` for `n` types of prompts).
For the case where a list of multiple prompts is given as a value of key `prompt`, one prompt is randomly sampled and sent during training.

3. Prepare LMDB
    Use the scipt `./CRIS.pytorch/prepare_datasets_all.sh`, by editing necessary paths.
For more reference, refer `./CRIS.pytorch/tools/prepare_datasets.md`.

4. Edit path arguments and prompt type in config
    Data: Paths to fill are commented
    Train: Change the following args 
        output_folder
        prompt_type

5. Run scripts
    a. For inference
    Use the script ./CRIS.pytorch/test.sh as sample and edit necessary paths.
    b. For finetunning
    Use the script ./CRIS.pytorch/finetune.sh as sample and edit necessary paths.

### **Running CLIPSeg**

### Zero Shot Segmentation

To perform zero-shot segmentation, you can use the provided script. Open a terminal and navigate to the project directory, then execute the following command:
```bash
bash ./VL-SEG/scripts/zss.sh
```
This script will initiate the zero-shot segmentation process and produce the desired results.

### Fine-Tuning

If you need to run fine-tuning for your model, you can do so using the following script:
```bash
bash ./VL-SEG/scripts/finetune.sh
```
This script will start the fine-tuning process, which is essential for customizing the model for specific tasks.

## Dataset used in our experiments
##### (Some of the datasets may requires requesting permissions to get access to the datasets)
- [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
	- This dataset can be downloaded directly from the given link by navigating to the download section.
- [ClinicDB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)
	- This dataset can be downloaded directly from the given link.
- [BKAI](https://www.kaggle.com/datasets/magiccard/bkai-igh)
	- This dataset can be downloaded directly from the given link.
- [CVC-300](https://github.com/DengPingFan/PraNet)
	- This dataset can be downloaded directly from the given link by navigating to the section 3.1.2 of README file(link to download the zip file is available). 
- [CVC-ColonDB](https://github.com/DengPingFan/PraNet)
	- This dataset can be downloaded directly from the given link by navigating to the section 3.1.2 of README file(link to download the zip file is available). 
- [ETIS](https://github.com/DengPingFan/PraNet)
	- This dataset can be downloaded directly from the given link by navigating to the section 3.1.2 of README file(link to download the zip file is available). 
- [ISIC-2016](https://challenge.isic-archive.com/data/)
	- This dataset can be downloaded directly from the given link.
- [DFU-2022](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php)
	- This dataset can be downloaded by filling the form whose link is available in the license agreement which can be found on the `Diabetic Foot Ulcers Datasets` section of the table present in the `Datasets/Software` section of the given link.
- [CAMUS](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8)
	- This dataset can be downloaded directly from the given link by logging in to the site. 
- [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
	- This dataset can be downloaded directly from the given link.
- [CheXlocalize](https://github.com/rajpurkarlab/cheXlocalize#download)
	- This dataset can be downloaded by following the instructions from the given link.

## License
	Details will be added soon.
## Citation
	Details will be added soon.
