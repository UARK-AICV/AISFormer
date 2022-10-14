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
    # print(f'n polygons: {len(polygons)}')
    formatted_polygons = []
    for p in polygons:
        formatted_polygons.append((p[0], p[1]))

    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(formatted_polygons, outline=1, fill=1)
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
    ANNOT_FILE = '/home/tqminh/AmodalSeg/data/std_data/COCOA/annotations/instances_train2014.json'
    IMG_DIR = '/home/tqminh/AmodalSeg/data/std_data/COCOA/train2014'
    coco = COCO(ANNOT_FILE)
    cat_ids = coco.getCatIds() 

    #targetImg = 'COCO_val2014_000000019663.jpg'
    #targetImg = 'COCO_val2014_000000032538.jpg'

    #new ones
    #targetImg = 'COCO_val2014_000000237399.jpg' # meo (ann[0]) va laptop (ann[2])
    #targetImg = 'COCO_val2014_000000328805.jpg'

    #train ones
    #targetImg = 'COCO_train2014_000000020818.jpg' # occludee (1), occluder (0)
    #targetImg = 'COCO_train2014_000000021313.jpg' # occludee (1), occluder (0)
    targetImg = 'COCO_train2014_000000018396.jpg' # occludee (1), occluder (0)

    occluder_id = 0
    occludee_id = 1
 

    for coco_img_key in coco.imgs.keys():
        if coco.imgs[coco_img_key]['file_name'] == targetImg:
            img_info = coco.loadImgs(coco_img_key)[0]
            annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=[], iscrowd=None)
            anns = coco.loadAnns(annIds)

            # cv_img = cv2.imread(img_info['file_name'])
            # cv_img = cv2.imread('{}/{}'.format(IMG_DIR,img_info['file_name']), cv2.COLOR_RGB2BGR)
            cv_img = read_image('{}/{}'.format(IMG_DIR,img_info['file_name']), 'RGB')
            visualizer1 = Visualizer(cv_img)
            visualizer2 = Visualizer(cv_img)
            visualizer3 = Visualizer(cv_img)
            visualizer4 = Visualizer(cv_img)

            H, W, C = cv_img.shape
            #for ann in anns:

            ann = anns[occludee_id]
            segm = ann.get("segmentation", None)
            i_segm = ann.get("inmodal_seg", None)
            bbox = ann.get("bbox", None)

            ann_occluder = anns[occluder_id]
            segm_occluder = ann_occluder.get("segmentation", None)
            occluder_mask = decode_rle_mask(segm_occluder)

            box_mask = create_box_mask(bbox, H, W)
            occluder_mask_in_box = box_mask & occluder_mask
            occluder_mask_in_box = occluder_mask_in_box.astype(bool)


            bbox = convert_box(bbox)

            boxes = Boxes(torch.tensor([bbox]))
            amodal_mask = decode_rle_mask(segm)
            inmodal_mask = decode_rle_mask(i_segm)

            invisible_mask = amodal_mask ^ inmodal_mask

            amodal_mask = np.array([amodal_mask])
            inmodal_mask = np.array([inmodal_mask])
            occluder_mask_in_box = np.array([occluder_mask_in_box])
            invisible_mask = np.array([invisible_mask])
            

            visualizer1.draw_mask_only(amodal_mask, boxes, colors=[np.array([0., 0., 1.], dtype=np.float32)], name='amodal')
            visualizer2.draw_mask_only(inmodal_mask, boxes, colors=[np.array([0., 1., 0.], dtype=np.float32)], name='inmodal')
            visualizer3.draw_mask_only(occluder_mask_in_box, boxes, colors=[np.array([1., 0., 0.], dtype=np.float32)], name='occluder')
            visualizer4.draw_mask_only(invisible_mask, boxes, colors=[np.array([0., 1., 1.], dtype=np.float32)], name='invisible')
     

            print()

main()
