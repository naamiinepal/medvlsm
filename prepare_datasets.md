## Prepare datasets

Follow the instructions in this file to prepare the datasets.

### 1. Download dataset

#### Dataset used in our experiments

**NOTE: (Some of the datasets may require requesting permissions to get access to the datasets))

- [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
	- This dataset can be downloaded directly from the given link by navigating to the download section.
- [ClinicDB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)
	- This dataset can be downloaded directly from the given link.
- [BKAI](https://www.kaggle.com/datasets/magiccard/bkai-igh)
	- This dataset can be downloaded directly from the given link.
- [CVC-300](https://github.com/DengPingFan/PraNet)
	- This dataset can be downloaded directly from the given link by navigating to section 3.1.2 of the README file (a link to download the zip file is available). 
- [CVC-ColonDB](https://github.com/DengPingFan/PraNet)
	- This dataset can be downloaded directly from the given link by navigating to section 3.1.2 of the README file (a link to download the zip file is available). 
- [ETIS](https://github.com/DengPingFan/PraNet)
	- This dataset can be downloaded directly from the given link by navigating to section 3.1.2 of the README file (a link to download the zip file is available). 
- [ISIC-2016](https://challenge.isic-archive.com/data/)
	- This dataset can be downloaded directly from the given link.
- [DFU-2022](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php)
	- This dataset can be downloaded by filling out the form whose link is available in the license agreement, which can be found in the `Diabetic Foot Ulcers Datasets` section of the table in the `Datasets/Software` section of the given link.
- [CAMUS](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8)
	- This dataset can be downloaded directly from the given link by logging in to the site. 
- [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
	- This dataset can be downloaded directly from the given link.
- [CheXlocalize](https://github.com/rajpurkarlab/cheXlocalize#download)
	- Follow the instructions in the **Download Data** section.
	- The dataset contains mask information in JSON files. Use the [scripts/chexlocalize_json_to_mask.py](scripts/chexlocalize_json_to_mask.py) file to convert this JSON segmentation to a binary image mask.

### 2. Prepare the data splits JSON (containing prompts)

For all the datasets (except ISIC and CAMUS), we prepared the data splits (train-val-test) as these splits were not available in the original dataset.

  For all the datasets, structure the downloaded datasets such that the `images` and `masks` directories are inside the same folder. 
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
  
  Create the data splits JSON files (containing image name, mask name, and all prompts for individual image-mask pair) by using the [scripts/get_data_json.py](scripts/get_data_json.py) file.
  
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
    Here, the prompts *p5* and *p6* have not been included as these are extracted from the pathology provided by the [CheXpert dataset](https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c).
    Please download the `CSV` files in the link and extract the pathologies.
    For simplicity, you can prepare a CSV file containing the filename and corresponding pathologies, then change the above script.
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

After running these scripts, the data split JSON files will be structured as shown below.
```
├── output_data_dir
      ├── testA.json
      ├── testB.json
      ├── train.json
      └── val.json
```

Each JSON file will have a list of dictionaries, and each dictionary is structured as shown below.
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
In the key prompts of each annotation, there should be a key-value pair of prompts of each type (`P0, P1, ..., Pn` for `n` types of prompts). 
When a list of multiple prompts is given as a key `prompt` value, one prompt is randomly sampled and sent during training.

Generating these JSON split files is sufficient to run the experiments for CLIPSeg.

### 3. Prepare lmdb files for CRIS experiments

Our experiment for CRIS is a fork of the [original CRIS implementation](https://github.com/DerrickWang005/CRIS.pytorch).
So, to prepare the data pipeline compatible with this implementation, we generate lmdb files and structure the datasets
following the [instructions provided in their implementation](https://github.com/DerrickWang005/CRIS.pytorch/blob/master/tools/prepare_datasets.md)

- **LMDB for all non-radiology datasets**

  To prepare lmdb files for all the non-radiology datasets, run the `./CRIS.pytorch/prepare_datasets_all.sh` by editing the necessary paths.

- **LMDB for CheXlocalize dataset**

  To prepare lmdb files for the CheXlocalize dataset, run the [scripts/folder2lmdb_chexlocalize.py](scripts/folder2lmdb_chexlocalize.py) file.

- **LMDB for CAMUS and BUSI**

  To prepare lmdb files for CAMUS and BUSI datasets, run `CRIS.pytorch/prepare_camus_datasets.sh` and `CRIS.pytorch/prepare_busi_datasets.sh` respectively.

### 4. Final dataset folder structure

Create a `datasets` folder, and structure the prepared datasets in the folder structure as shown below.

```
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

