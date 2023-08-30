## Prepare datasets

Follow the instructions in this file to prepare the datasets.

### 1. Download dataset

Download the datasets following the instructions in the [here](README.md##dataset-used-in-our-experiments).

### 2. Prepare the data splits json (containing prompts)

For all the datasets (except ISIC and CAMUS), we prepare the data splits (train-val-test) splits as these splits were not available in the orginal dataset.

  For all the datasets, structure the downloaded datasets such that `images` and `masks` directories are inside the same folder. 
  ```
  ├── dataset_root_dir
      ├── images
      |   ├── file_1.png
      |   └── file_n.png
      └── masks
          ├── file_1.png
          └── file_n.png
  ```
#### a. For non-radiology datasets
  
  Create the data splits json files (containing image name, mask name and the all prompts for individual image-mask pair) by using the [scripts/get_data_json.py](scripts/get_data_json.py) file.
  
  ```shell
  python scripts/get_data_json.py \
    --root_data_dir <dataset_root_dir> \
    --dataset_name <dataset_name> \
    --image_dir <images_dir_relative_to_root_dir> \
    --mask_dir <masks_dir_relative_to_root_dir> \
    --valid_per <val_percent> \
    --test_per <test_percent> \
    --output_data_dir <output_dir>
  ```

#### b. For radiology datasets

- **CheXlocalize**
  
    To generate data splits for this dataset, run the [scripts/get_chexlocalize_json.py](scripts/get_chexlocalize_json.py) file.
    ```shell
    python scripts/get_chexlocalize_json.py \
      --image-dir <image_dir> \
      --mask-dir <mask_dir>
      --test-ratio 0.2 \
      --val-ratio 0.2 \
      --verbose \
      --output-dir <output_dir>
    ```
    Here the prompts *p5* and *p6* has not been included as these are extracted from the pathology provided from the [CheXpert dataset](https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c).
    Please download the `csv` files for provided in the link and extract the pathologies.
    For simplicity, you can prepare a csv file containing the filename and corresponding pathologes, then change the above script.
    Add new templates for *p5* AND *p6* as:
     ```python
      p5_template = convert_str_to_template(
        "$labels of shape $shape, and located in $location of the $xray_view of a Chest Xray. $pathologies are present."
      )
  
     p6_template = convert_str_to_template(
       "$labels in a Chest Xray. $pathologies are present."
     )
     ```

- **CAMUS**
    
    To generate the prompts and data splits for CAMUS, run the [scripts/get_camus_json.py](scripts/get_camus_json.py) file.
    ```shell
    python scripts/get_camus_json.py \
      --img-dir <image_dir> --img-pattern <image_files_pattern> \
      --mask-dir <mask_dir> --mask-pattern <mask_file_pattern> \
      --output-dir <output-dir> \
      --val-ratio <val-ratio> \
      --test-ratio <test_ratio> \
    ```

- **BUSI**

  To generate the prompts and data splits for BUSI, run the [scripts/get_bus_json.py](scripts/get_bus_json.py) file.
  ```shell
  python scripts/get_bus_json.py \
    --root-dir <dataset_root_dir> \
    --image-glob <image_glob_pattern> \
    --masks-glob <masks_glob_pattern> \
    --val-ratio <val_ratio> \
    --test-ratio <test_ratio> \
    --output-dir <output_dir>
  ```

After running these scripts, the data split json files will be structured as shown below.
```
├── output_data_dir
      ├── testA.json
      ├── testB.json
      ├── train.json
      └── val.json
```

Each json file will have a list of dictionaries, and each dictionary is structured as shown below.
```json
{
        "bbox": [
            419,
            492,
            541,
            743
        ],
        "cat": 0,
        "segment_id": 930,
        "img_name": "930.jpg",
        "mask_name": "930.png",
        "sentences": [
            {
                "idx": 0,
                "sent_id": 958,
                "sent": ""
            }
        ],
        "prompts": {
            "p0": "",
            "p1": "polyp",
            "p2": "square polyp",
            "p3": "black square polyp",
            "p4": "small black square polyp",
            "p5": "one small black square polyp",
            "p6": "one small black square polyp, located in center of the image",
        },
        "sentences_num": 1
    }
```
In the key prompts of each of the annotation, there should be key value pair of prompts of each type (`P0, P1, ..., Pn` for `n` types of prompts). 
For the case where a list of multiple prompts is given as a value of key `prompt`, one prompt is randomly sampled and sent during training.

Generating these json split files is sufficient to run the experiments for CLIPSeg.

### 3. Prepare lmdb files for CRIS experiments

Our experiments for CRIS is a fork of the [original CRIS implementation](https://github.com/DerrickWang005/CRIS.pytorch).
So, in order to prepare the data pipeline compatible with this implementation, we generate lmdb files and structure the datasets
following the [instructions provided in their implementation](https://github.com/DerrickWang005/CRIS.pytorch/blob/master/tools/prepare_datasets.md)

#### a. LMDB for all non-radiology datasets

  To prepare lmdb files for all the non-radiology datasets, run the `./CRIS.pytorch/prepare_datasets_all.sh` by editing the necessary paths.

#### b. LMDB for CheXlocalize dataset

  To prepare lmdb files for CheXlocalize dataset, run the [scripts/folder2lmdb_chexlocalize.py](scripts/folder2lmdb_chexlocalize.py) file.

### 2. Datasets struture

After the above-mentioned commands, the strutre of the dataset folder should be like:

```none
├── anns
│   ├── bkai_polyp
│   │   ├── testA.json
│   │   ├── testB.json
│   │   ├── train.json
│   │   └── val.json
│   └── other_datasets
│       ├── testA.json
│       ├── testB.json
│       ├── train.json
│       └── val.json
├── images
|   ├── bkai_polyp
|   |   ├── file_1.png
|   |   └── file_n.png
|   └── other_datasets
|       ├── file_1.png
|       └── file_n.png
├── lmdb
│   ├── bkai_polyp
│   │   ├── testA.lmdb
│   │   ├── testB.lmdb
│   │   ├── train.lmdb
│   │   └── val.lmdb
│   └──  other_datasets
│       ├── testA.lmdb
│       ├── testB.lmdb
│       ├── train.lmdb
│       └── val.lmdb    
└── masks
    ├── bkai_polyp
    |   ├── file_1.png
    |   └── file_n.png
    └── other_datasets
        ├── file_1.png
        └── file_n.png

