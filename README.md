# Exploring Transfer Learning in Medical Image Segmentation using Vision-Language Models



## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [Zero Shot Segmentation](#zero-shot-segmentation)
  - [Fine-Tuning](#fine-tuning)

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

## Usage

### Dataset Preparation
Before running any experiments, you need to ensure that the provided dataset is correctly placed within the `data` folder at the root of the project. The directory structure of the `data` folder should look like this:
```
data/
│
├── bkai_polyp/
│   ├── anns/
│   │   ├── test.json
│   │   ├── train.json
│   │   └── val.json
│   ├── images/
│   └── masks/
│
├── [other dataset folders...]
│
└── kvasir_polyp/
    ├── anns/
    │   ├── test.json
    │   ├── train.json
    │   └── val.json
    ├── images/
    └── masks/
```
Each dataset folder (`bkai_polyp`, `busi`, `camus`, etc.) contains three sub-directories: `anns`, `images`, and `masks`. The anns directory contains prompt files (`test.json`, `train.json`, `val.json`), while `images` and `masks` hold input images and target masks respectively.

### Zero Shot Segmentation

To perform zero-shot segmentation, you can use the provided script. Open a terminal and navigate to the project directory, then execute the following command:
```bash
bash scripts/zss.sh
```
This script will initiate the zero-shot segmentation process and produce the desired results.

### Fine-Tuning

If you need to run fine-tuning for your model, you can do so using the following script:
```bash
bash scripts/finetune.sh
```
This script will start the fine-tuning process, which is essential for customizing the model for specific tasks.
