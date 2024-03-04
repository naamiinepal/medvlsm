import json
import random
from argparse import ArgumentParser

random.seed(444)

sample_fracs=[0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]

def main(annotations_root):

    with open(f"{annotations_root}]/anns/train.json") as train_data:
        total_data = json.load(train_data)
    
    print(total_data[0])
    random.shuffle(total_data) 
    
    print(total_data[0])
    len_total = len(total_data)
    
    for frac in sample_fracs:
        frac_data = total_data[0:int(frac*len_total)]
    
        with open(f"{annotations_root}]/anns_random_sampled/train_{frac}.json", "w") as op:
            op.write(json.dumps(frac_data))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--annotations_root",
        type=str,
        default=None,
        help="root directory where the anns directory for the particular dataset is stored",
    )
    args = parser.parse_args()

    main(**vars(args))
