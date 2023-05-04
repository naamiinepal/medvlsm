import glob

import cv2
import numpy as np
from num2words import num2words
from PIL import Image
from scipy import ndimage
from skimage.io import imread, imshow
from skimage.measure import find_contours, label, regionprops

""" Convert a mask to border image """


def mask_to_border(mask):

    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border


""" Mask to bounding boxes """


def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes


""" Mask to bounding features """


def mask_to_overall_bbox(mask_path):
    """_summary_
    function to get auxiliary information of image
    Args:
        mask_path (string): mask image's path

    Returns:
        list: a list of overall bbox coordinates

    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    bboxes = mask_to_bbox(mask)
    # num_polyps = 0 if len(bboxes) == 1 else 1
    # polyp_sizes = None
    min_x1 = mask.shape[1]
    min_y1 = mask.shape[0]
    max_x2 = 0
    max_y2 = 0
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if float(area) > 4:
            if x1 < min_x1:
                min_x1 = x1
            if y1 < min_y1:
                min_y1 = y1
            if x2 > max_x2:
                max_x2 = x2
            if y2 > max_y2:
                max_y2 = y2
    return [min_x1, min_y1, max_x2, max_y2]


def patch_coverage(mask):
    """_summary_
    function to get auxiliary information of image
    Args:
        mask_path (string): mask image's path
    Returns:
        dict: a dict of description about realtive area of mask in image and its spread across four quadrants
    """
    h, w = mask.shape
    total_pix = np.shape(mask)[0] * np.shape(mask)[1]
    num_white_pix = np.sum(mask == 1)
    coverage = num_white_pix / total_pix

    first_quad_white_pix = np.sum(mask[0 : int(h / 2), 0 : int(w / 2)] == 1)
    sec_quad_white_pix = np.sum(mask[0 : int(h / 2), int(w / 2) :] == 1)
    third_quad_white_pix = np.sum(mask[int(h / 2) :, 0 : int(w / 2)] == 1)
    fourth_quad_white_pix = np.sum(mask[int(h / 2) :, int(w / 2) :] == 1)

    first_per = first_quad_white_pix / num_white_pix
    sec_per = sec_quad_white_pix / num_white_pix
    third_per = third_quad_white_pix / num_white_pix
    fourth_per = fourth_quad_white_pix / num_white_pix
    return {
        "coverage": coverage,
        "per_quad": {
            "first_per": first_per,
            "sec_per": sec_per,
            "third_per": third_per,
            "fourth_per": fourth_per,
        },
    }


def get_mask_decription(mask_path):
    """_summary_
    function to get auxiliary information of image
    Args:
        mask_path (string): mask image's path
    Returns:
        _type_: _description_
    """
    separate_masks = []
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = np.array((mask / 255.0) > 0.5, dtype=int)

    # get masks labelled with different values
    label_im, nb_labels = ndimage.label(mask)
    res = {}

    j = 0
    sizes = []
    for i in range(nb_labels):
        mask_compare = np.full(np.shape(label_im), i + 1)
        # check equality test and have the value 1 on the location of each mask
        separate_mask = np.equal(label_im, mask_compare).astype(int)

        res_i = patch_coverage(separate_mask)
        if res_i["coverage"] > 0.001:
            res[i] = res_i
            j = j + 1
            if res_i["coverage"] > 0.30:
                sizes.append("large")
            elif res_i["coverage"] > 0.08:
                sizes.append("medium")
            elif res_i["coverage"] < 0.01:
                sizes.append("tiny")
            else:
                sizes.append("small")

    tiny_polyp = sizes.count("tiny")
    small_polyp = sizes.count("small")
    medium_polyp = sizes.count("medium")
    large_polyp = sizes.count("large")

    sizes_str = "skin cancer"
    # sizes_str = "In colon polyp is an oval bump, often in pink color, "

    # if tiny_polyp > 0:
    #    sizes_str = sizes_str + num2words(tiny_polyp) + " tiny sized polyps, "
    # if small_polyp > 0:
    #    sizes_str = sizes_str + num2words(small_polyp) + " small sized polyps, "
    # if medium_polyp > 0:
    #    sizes_str = sizes_str + num2words(medium_polyp) + " medium sized polyps, "
    # if large_polyp > 0:
    #    sizes_str = sizes_str + num2words(large_polyp) + " large sized polyps"

    return res, sizes_str.strip().strip(",")
