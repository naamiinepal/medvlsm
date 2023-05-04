import json
import os
import shutil


def extract():
    with open("/mnt/Enterprise/miccai_2023_CRIS/CRIS.pytorch/datasets/anns/kvasir_polyp/testA.json") as of:
        op_dir = "/mnt/Enterprise/miccai_2023_CRIS/CRIS.pytorch/datasets/kvasir_val"
        ip_dir = "/mnt/Enterprise/PUBLIC_DATASETS/polyp_datasets/Kvasir-SEG"
        op = json.load(of)

        for img in op:
            id = img["segment_id"]
            src_img = os.path.join(ip_dir, "images_cf", str(id)+".jpg")
            src_mask = os.path.join(ip_dir, "masks_cf", str(id)+".png")
            dest_img = os.path.join(op_dir, "images", str(id)+".jpg")
            dest_mask = os.path.join(op_dir, "masks", str(id)+".png") 
            shutil.copyfile(src_img, dest_img)
            shutil.copyfile(src_mask, dest_mask)
            print(img['segment_id'])
extract()