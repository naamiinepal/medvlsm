## Dataset Preparation

### Dataset Sources

- [BKAI](https://www.kaggle.com/c/bkai-igh-neopolyp/data)
- [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- [CAMUS](http://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8)
- [CheXlocalize](https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c)
- [ClinicDB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)
- [ColonDB](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-colondb)
- [CVC300](https://drive.google.com/drive/folders/10QXjxBJqCf7PAXqbDvoceWmZ-qF07tFi?usp=sharing)
- [ETIS](https://drive.google.com/drive/folders/10QXjxBJqCf7PAXqbDvoceWmZ-qF07tFi?usp=sharing)
- [DFU](https://dfu-challenge.github.io/dfuc2022.html)
- [ISIC](https://challenge.isic-archive.com/data/)
- [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)

### Dataset Structure and Processing
Before running any experiments, you need to ensure that the provided dataset is correctly placed within the `data/` folder at the root of the project.
The directory structure of the `data/` folder should look like this:
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

Each dataset folder (`bkai_polyp`, `busi`, `camus`, etc.) contains three sub-directories: `anns/`, `images/`, and `masks/`. 
The anns directory contains prompt files (`test.json`, `train.json`, `val.json`), while `images/` and `masks/` hold input images and target masks respectively.
The name of the mask files should be consistent with the `mask_name` value given in the `*.json` files; if not, you may need to rename them accordingly.
All of the masks must be binary segmentation images with `0` for background and `255` for foreground; if not, please process them accordingly.

**Note:** *Since the datasets are of medical domain, you may need to comply with certain conditions before downloading and using them.* 