from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import imantics
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tools.vis_gt import IMG_DIR
from skimage.transform import resize
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.detection_utils import read_image
from detectron2.structures import Boxes
import torch
import os
from os.path import join


def from_rle_str_to_plg(encode_str):
    binary_mask = maskUtils.decode(encode_str)
    polygons = imantics.Mask(binary_mask).polygons()

    return [l.tolist() for l in polygons]

def decode_rle_mask(encode_str):
    return maskUtils.decode(encode_str)

def create_box_mask(box, img_h, img_w):
    result = np.full((img_h, img_w), False)
    x, y, h, w = box
    x, y, h, w = int(x), int(y), int(h), int(w)
    result[y:y+w, x:x+h ] = True

    return result

def get_binary_mask(polygons, height, width):
    img = Image.new('L', (width, height), 0)
    for polygon in polygons:
        formatted_polygon = []
        for i in range(0, len(polygon)-1, 2):
            formatted_polygon.append((polygon[i], polygon[i+1]))

        #print(polygon)
        #print(formatted_polygon)
        ImageDraw.Draw(img).polygon(formatted_polygon, outline=1, fill=1)

    mask = np.array(img)
    return mask

def convert_box(box):
    #import pdb;pdb.set_trace()
    x, y, h, w = box
    x1 = x
    x2 = x1 + h
    y1 = y
    y2 = y1 + w

    return [x1, y1, x2, y2]

def to_3(mask):
    img = np.zeros((mask.shape[0], mask.shape[1], 3))
    img[:, :, 0] = mask
    img[:, :, 1] = mask
    img[:, :, 2] = mask

    return img



def main():
    # cocoa
    ANNOT_FILE = '/home/tqminh/AmodalSeg/data/std_data/COCOA/annotations/instances_train2014.json'
    IMG_DIR = '/home/tqminh/AmodalSeg/data/std_data/COCOA/selected'
    OUTPUT_DIR = '/home/tqminh/AmodalSeg/data/outtest/visualize_GT_cocoa/'

    # kins
    # ANNOT_FILE = '/home/tqminh/AmodalSeg/data/std_data/KINS/annotations/instances_val2017.json'
    # IMG_DIR = '/home/tqminh/AmodalSeg/data/std_data/KINS/selected'
    # OUTPUT_DIR = '/home/tqminh/AmodalSeg/data/outtest/visualize_GT_kins/'

    # d2sa
    # ANNOT_FILE = '/home/tqminh/AmodalSeg/data/std_data/D2SA/annotations/instances_val2017.json'
    # IMG_DIR = '/home/tqminh/AmodalSeg/data/std_data/D2SA/selected/'
    # OUTPUT_DIR = '/home/tqminh/AmodalSeg/data/outtest/visualize_GT_d2sa/'

    coco = COCO(ANNOT_FILE)
    cat_ids = coco.getCatIds()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for coco_img_key in coco.imgs.keys():
        img_info = coco.loadImgs(coco_img_key)[0]

        if img_info['file_name'] != 'COCO_train2014_000000019838.jpg':
            continue

        try:
            cv_img = read_image('{}/{}'.format(IMG_DIR,img_info['file_name']), 'RGB')
            print(img_info['file_name'])
        except:
            print("the folder does not contain the image")
            continue

        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=[], iscrowd=None)
        anns = coco.loadAnns(annIds)

        visualizer = Visualizer(cv_img)
        H, W, C = cv_img.shape
        amodal_masks = []
        for ann in anns:
            segm = ann.get("segmentation", None)
            #segm = img_info['amodal_full']
            if type(segm) == list:
                amodal_masks.append(get_binary_mask(segm, H, W))
            else:
                amodal_masks.append(decode_rle_mask(segm))

        
        #amodal_masks[1], amodal_masks[2] = amodal_masks[2], amodal_masks[1]
        #amodal_masks[1], amodal_masks[0] = amodal_masks[0], amodal_masks[1]
        #amodal_masks = amodal_masks[11:15]

        amodal_masks = np.array(amodal_masks)
        my_colors = [
                     np.array([0., 1., 0.], dtype=np.float32),np.array([0. , 0., 1.], dtype=np.float32),
                      np.array([0., 1., 1.], dtype=np.float32),np.array([0.5, 1., 0.], dtype=np.float32),
                    np.array([1., 0., 0.], dtype=np.float32), np.array([0.5, 1., 0.], dtype=np.float32),
                    np.array([0.5, 0.5, 0.], dtype=np.float32)
                ]
        visualizer.draw_masks(
                amodal_masks, name=join(OUTPUT_DIR, img_info['file_name']),
                colors=my_colors
            )
        print()


main()
