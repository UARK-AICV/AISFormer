import os 
import os.path as osp 
import cv2 
from tqdm import tqdm
import numpy as np 
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import json


IMG_DIR = '/home/tqminh/AmodalSeg/data/std_data/D2SA/d2s_amodal_images_v1/images'
ANNOT_FILE = '/home/tqminh/AmodalSeg/data/std_data/D2SA/annotations/instances_train2017_new_coop.json'
SAVE_DIR = '/home/tqminh/AmodalSeg/data/std_data/D2SA/vis_gt'
COLOR_MAP = {}
BLEND_RATIO = 0.4
cats_to_vis = [3] 

os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
def from_cv_to_pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img) 

def from_pil_to_cv(im):
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

def get_binary_mask(polygons, height, width):
    # print(f'n polygons: {len(polygons)}')
    formatted_polygons = []
    for p in polygons:
        formatted_polygons.append((p[0], p[1]))

    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(formatted_polygons, outline=1, fill=1)
    mask = np.array(img)
    return mask

def get_color_given_id(obj_id: int):
    # if COLOR_MAP.get(obj_id) is not None:
    #     return COLOR_MAP[obj_id]

    #TODO: find more simple random
    step = 10
    r = int((0 + obj_id*step + obj_id*step*step/256)%256) #int(obj_id/(256**3))
    g = int((255*(obj_id%2) - obj_id*step + obj_id*step*step/256)%256)
    b = int((255*(obj_id+1)%2 + obj_id*step + obj_id*step*step/256)%256)
    COLOR_MAP[obj_id] = np.array([b, g, r])

    return np.array([b, g, r])

def from_cv_to_pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img) 

def from_pil_to_cv(im):
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

def main():
    coco = COCO(ANNOT_FILE)
    cat_ids = coco.getCatIds()
    # if cats_to_vis is None:
    #     cats_to_vis = cat_ids

    N_PRINT = 100
    count = 0
    for imgId in coco.imgs:
        if count > N_PRINT:
            break
        img_info = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=[], iscrowd=None)
        anns = coco.loadAnns(annIds)

        #cv_img = cv2.imread('{}/{}'.format(IMG_DIR,img_info['file_name']))
        cv_img = cv2.imread(img_info['file_name'])
        H, W, C = cv_img.shape


        draw_img = cv_img.copy() 
        is_draw = True
        for i, ann in enumerate(anns):
            ann_id = ann['id']
            # if ann['category_id'] != 1:
            #     continue
            bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
            centroid={
                'x': bbox_x + bbox_w / 2,
                'y': bbox_y + bbox_h / 2
            }
            box_area = bbox_w * bbox_h
            # seg = ann['inmodal_seg'][0]
            segs = ann['segmentation']
            for seg in segs:
                try:
                    polygon = np.array(seg).reshape((int(len(seg)/2), 2))
                    bin_mask = get_binary_mask(polygon, H, W)
                except:
                    continue
                polygon_color = get_color_given_id(ann_id)

                '''
                pil_img = from_cv_to_pil(draw_img)
                draw_img_pil = from_cv_to_pil(draw_img)
                draw_pil = ImageDraw.Draw(draw_img_pil)
                px = polygon[:,0].tolist()
                py = polygon[:,1].tolist()
                ps = []
                for pi in range(len(px)):
                    ps.append((px[pi],py[pi]))
                draw_pil.polygon(ps, fill=None, outline="red")
                
                vis_pil = Image.blend(pil_img, draw_img_pil, 0.5)
                draw_img = from_pil_to_cv(vis_pil)
                '''
                draw_img[np.where(bin_mask > 0)] = cv_img[np.where(bin_mask > 0)]*BLEND_RATIO + (1-BLEND_RATIO)*polygon_color  
            is_draw = True
            start_point = (int(bbox_x), int(bbox_y))
            end_point = (int(bbox_x + bbox_w), int(bbox_y + bbox_h))

            cv2.rectangle(draw_img, start_point, end_point, [0, 0, 255], 1)

        if is_draw:
            #cv2.imwrite(osp.join(SAVE_DIR, img_info['file_name']), draw_img)
            cv2.imwrite(osp.join(SAVE_DIR, os.path.basename(img_info['file_name'])), draw_img) 
            count += 1


if __name__ == '__main__':
    main()
