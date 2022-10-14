from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import imantics
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tools.vis_gt import IMG_DIR
from skimage.transform import resize
from colordict import *

def from_rle_str_to_plg(encode_str):
    binary_mask = maskUtils.decode(encode_str)
    polygons = imantics.Mask(binary_mask).polygons()

    return [l.tolist() for l in polygons]

def get_binary_mask(polygons, height, width):
    # print(f'n polygons: {len(polygons)}')
    formatted_polygons = []
    for p in polygons:
        formatted_polygons.append((p[0], p[1]))

    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(formatted_polygons, outline=1, fill=1)
    mask = np.array(img)
    return mask

def to_3(mask):
    img = np.zeros((mask.shape[0], mask.shape[1], 3))
    img[:, :, 0] = mask
    img[:, :, 1] = mask
    img[:, :, 2] = mask

    return img



def main():
    ANNOT_FILE = '/home/tqminh/AmodalSeg/data/std_data/COCOA/annotations/instances_val2014.json'
    IMG_DIR = '/home/tqminh/AmodalSeg/data/std_data/COCOA/val2014'
    coco = COCO(ANNOT_FILE)
    cat_ids = coco.getCatIds() 

    targetImg = 'COCO_val2014_000000019663.jpg'
    #targetImg = 'COCO_val2014_000000032538.jpg'
    fix_box = None

    for coco_img_key in coco.imgs.keys():
        if coco.imgs[coco_img_key]['file_name'] == targetImg:
            img_info = coco.loadImgs(coco_img_key)[0]
            annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=[], iscrowd=None)
            anns = coco.loadAnns(annIds)

            # cv_img = cv2.imread(img_info['file_name'])
            cv_img = cv2.imread('{}/{}'.format(IMG_DIR,img_info['file_name']))
            H, W, C = cv_img.shape
            for ann in anns:
                segm = ann.get("segmentation", None)
                i_segm = ann.get("inmodal_seg", None)
                bbox = ann.get("bbox", None)
                if fix_box == None:
                    fix_box = bbox
                try:
                    segm = from_rle_str_to_plg(segm)
                    i_segm = from_rle_str_to_plg(i_segm)
                except:
                    pass
                seg = segm[0]
                i_seg = i_segm[0]

                polygon = np.array(seg).reshape((int(len(seg)/2), 2))
                i_polygon = np.array(i_seg).reshape((int(len(i_seg)/2), 2))

                polygon[:, 0] -= int(fix_box[0])
                polygon[:, 1] -= int(fix_box[1])

                i_polygon[:, 0] -= int(fix_box[0])
                i_polygon[:, 1] -= int(fix_box[1])

                binmask = get_binary_mask(polygon, int(fix_box[3]), int(fix_box[2]))
                i_binmask = get_binary_mask(i_polygon, int(fix_box[3]), int(fix_box[2]))

                
                binmask = resize(binmask - i_binmask, (45,45))
                boolmask = binmask > 0.0008


                colors = ColorDict()
                colImage = np.zeros((45,45,3), dtype="uint8")
                # colImage[:,:,0] = boolmask * 255 # for red

                colImage = np.multiply(to_3(boolmask), colors['orange'])


                # black_pixels = np.where(
                #     (colImage[:, :, 0] == 0) & 
                #     (colImage[:, :, 1] == 0) & 
                #     (colImage[:, :, 2] == 0)
                # )
                # colImage[black_pixels] = [255, 255, 255]

                cv2.imwrite('../data/outtest/vis_mask.png', colImage)
                
                import pdb;pdb.set_trace()
                #plt.imsave('../data/outtest/vis_mask.png', binmask)
                #exit(0)

            print()

main()
