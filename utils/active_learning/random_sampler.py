import json
import random
from argparse import ArgumentParser

random.seed(444)

sample_fracs=[0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]

def main(ds_root,ds_name):

    with open(f"{ds_root}/{ds_name}/anns/train.json") as train_data:
        total_data = json.load(train_data)
    
    print(total_data[0])
    random.shuffle(total_data) 
    
    print(total_data[0])
    len_total = len(total_data)
    
    for frac in sample_fracs:
        frac_data = total_data[0:int(frac*len_total)]
    
        with open(f"{ds_root}/{ds_name}/anns_random_sampled/train_{frac}.json", "w") as op:
            op.write(json.dumps(frac_data))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ds_root",
        type=str,
        default=None,
        help="root directory where the datasets are stored",
    )
    parser.add_argument(
        "--ds_name",
        type=str,
        default=None,
        help="name of the dataset",
    )
    args = parser.parse_args()

    main(**vars(args))
