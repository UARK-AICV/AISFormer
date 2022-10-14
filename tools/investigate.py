from pycocotools.coco import COCO
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import skimage.io as io
import random
import os
from os.path import basename
import cv2
from matplotlib.collections import PatchCollection
from tqdm import tqdm
import numpy
from PIL import Image, ImageDraw

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
import pickle

from sklearn.cluster import KMeans

def draw_boxes_on_image(img, bboxes, image_name):
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)

    cv2.imwrite("../data/outtest/" + image_name + ".jpg", img)

def draw_polygon_on_image(img_file_name, polygons):
    img = Image.open(img_file_name).convert('RGBA')

    img2 = img.copy()
    for polygon in polygons[1:]:
        formatted_polygon = []
        for p in polygon.tolist():
            formatted_polygon.append((p[0], p[1]))

        draw = ImageDraw.Draw(img2)
        draw.polygon(formatted_polygon, fill = "wheat")

    img3 = Image.blend(img, img2, 0.5)
    img3.save('../data/outtest/ok_out.png')


def get_binary_mask(polygons, height, width):
    img = Image.new('L', (width, height), 0)
    for polygon in polygons:
        formatted_polygon = []
        for p in polygon:
            formatted_polygon.append((p[0], p[1]))
        ImageDraw.Draw(img).polygon(formatted_polygon, outline=1, fill=1)

    mask = np.array(img)
    return mask

def get_cropped_mask(cv_img, polygons, bbox):
    x, y, w, h = bbox
    mask = get_binary_mask(polygons, cv_img.shape[0], cv_img.shape[1])
    polygon_cropped_img = cv2.bitwise_and(cv_img, cv_img, mask=mask)

    polygon_cropped_img = polygon_cropped_img[int(y): int(y + h), int(x): int(x + w)]
    polygon_cropped_mask = mask[int(y): int(y + h), int(x): int(x + w)]


    return np.uint8(255*polygon_cropped_mask)

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

if __name__ == "__main__":
    # dataDir='../data/chicken_data/syn_2_chickens'
    SHAPE = 28
    dataDir='/data/tqminh/AmodalSeg/std_data/KINS_2'
    dataType='train'
    annFile='{}/annotations/instances_{}2017.json'.format(dataDir,dataType)
    #annFile='{}/annotations/instances_{}_2017_transform_slight_correct.json'.format(dataDir,dataType)

    # Initialize the COCO api 
    coco=COCO(annFile)
    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)
    
    #filterClasses = ['chicken']
    #catIds = coco.getCatIds(catNms=filterClasses)
    #catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("Number of images containing all the  classes:", len(imgIds))

    n_anns = []
    overlap_ratios = []
    masks = []
    cnt_cats = {}
    
    for img_id in tqdm(range(1, len(imgIds) + 1)): # ids runs from 1
        # if img_id >=1000:
        #     break
        img_info = coco.loadImgs(imgIds[img_id - 1])[0]
        img = cv2.imread('{}/{}2017/{}'.format(dataDir,dataType,basename(img_info['file_name'])))

        #annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
        annIds = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        n_anns.append(len(anns))

        # get seg anns
        ax = plt.gca()
        polygons = []
        color = []
        bboxes = []
        b = []

        for i, ann in enumerate(anns):
            [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
            bboxes.append({
                'x1': bbox_x,
                'x2': bbox_x+bbox_w,
                'y1': bbox_y,
                'y2': bbox_y+bbox_h
            })
            b.append([bbox_x, bbox_y, bbox_x+bbox_w, bbox_y+bbox_h])
            if 'segmentation' in ann:
                # polygon
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg)/2), 2))
                    polygons.append(poly)

                    ax.add_patch(Polygon(polygons[i]))
                    ax.relim()
                    ax.autoscale_view()

                    plt.gca().invert_yaxis()
                    plt.axis('off')
                    canvas = ax.figure.canvas
                    canvas.draw()
                    # mask = np.array(canvas.renderer.buffer_rgba())[:,:,1]
                    # mask = cv2.resize(mask, (SHAPE, SHAPE))
                    ax.clear()

                    # cv2.imwrite(os.path.join(dataDir, 'masks/' \
                    #             + img_info['file_name'] + '_' + str(i)\
                    #                 + '.jpg'), mask)
                    
                    # check categories

                mask = get_cropped_mask(img, polygons, ann['bbox'])
                mask = cv2.resize(mask, (SHAPE, SHAPE))
                masks.append(mask)

                if str(ann['category_id']) not in cnt_cats.keys():
                    cnt_cats[str(ann['category_id'])] = 1
                else:
                    cnt_cats[str(ann['category_id'])] += 1

        # assert len(polygons) >= 2
        # print(img_info['file_name'])
        # draw_boxes_on_image(img, b, 'ok_out')
        # draw_polygon_on_image('{}/{}2017/{}'.format(
        #                             dataDir, dataType,
        #                             basename(img_info['file_name'])),
        #                         polygons)
        #if img_id==81 or '370553_77740.png' in img_info['file_name']:
        '''
        if img_id==37:
            print(img_info['file_name'])
            exit(0)
        '''

        for i in range(len(bboxes)):
            for j in range(i+1, len(bboxes)):
                if abs(get_iou(bboxes[i], bboxes[j]) - 0.0) > 0.01:
                    overlap_ratios.append(get_iou(bboxes[i], bboxes[j]))

    masks = np.array(masks)
    # import pdb;pdb.set_trace()
    f = open(os.path.join(dataDir, 'shape_priors_{}.pkl'.format(str(SHAPE))), 'wb')
    pickle.dump(masks, f)
    f.close()

    print("Mean box per frames: ", np.mean(n_anns))
    print("Mean overlap ratio: ", np.mean(overlap_ratios))
    print("Categories distribution: ", cnt_cats)
