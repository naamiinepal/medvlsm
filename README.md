## **SCRIPTS TO RUN CRIS**

1. Get this repository in your system.

2. Dataset preparation:
First of all, download the required datasets using the references given in the paper. Some of the datasets may require requesting permissions so please follow accordingly.
After downloading the dataset, prepare the dataset in Ref-COCO format.
The annotation format used in our experiments is slightly different from than original annotation format.
For reference format used in our experiment refer sample at `./CRIS.pytorch/datasets/anns/kvasir_polyp_80_10_10/train.json`(also see `val.json`, `testA.json` and `testB.json`).
In the key `prompts` of each of the annotations, there should be a key-value pair of prompts of each type (`P0, P1, ..., Pn` for `n` types of prompts).
For the case where a list of multiple prompts is given as a value of key `prompt`, one prompt is randomly sampled and sent during training.

3. Prepare the virtual environment.
Make sure python verison >= 3.6 is installed in the system and use the following command to create virtual environment.
```
python3 -m venv .venv
```
Activate the command.
```
source .venv/bin/activate
```
Install the required dependencies.
```
pip install -r requirements.txt
```

4. Prepare LMDB
    Use the script `./CRIS.pytorch/prepare_datasets_all.sh`, by editing necessary paths.
For more reference, refer `./CRIS.pytorch/tools/prepare_datasets.md`.

5. Edit path arguments and prompt type in config
    Data: Paths to fill are commented
    Train: Change the following args 
        output_folder
        prompt_type

6. Run scripts
    a. For inference
    Use the script ./CRIS.pytorch/test.sh as a sample and edit necessary paths.
    b. For finetuning
    Use the script ./CRIS.pytorch/finetune.sh as a sample and edit necessary paths.

### Dataset used in our experiments (Some of the datasets may require requesting permissions to get access to the datasets)
- [Kvasir-Seg](https://datasets.simula.no/kvasir-seg/)
- [ClinicDB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)
- [BKAI](https://www.kaggle.com/datasets/magiccard/bkai-igh)
- [CVC-300](https://github.com/DengPingFan/PraNet)(Navigate to section 3.1.2)
- [CVC-ColonDB](https://github.com/DengPingFan/PraNet)(Navigate to section 3.1.2)
- [ETIS](https://github.com/DengPingFan/PraNet)(Navigate to section 3.1.2)
- [ISIC-2016](https://challenge.isic-archive.com/data/)
- [DFU-2022](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php)
- [CAMUS]()
- [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- [CheXlocalize]()

