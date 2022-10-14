import os, json
from os.path import join


def proc_vis_mask_iseg(data_json):
    for ann in data_json['annotations']:
        ann['inmodal_seg'] = ann.pop('visible_mask')

    return data_json


if __name__=='__main__':
    ann_path = {
                'train': ['/home/tqminh/AmodalSeg/data/std_data/COCOA/COCOA_annotations_detectron/COCO_amodal_train2014_with_classes.json',
                            '/home/tqminh/AmodalSeg/data/std_data/COCOA/annotations/instances_train2014.json'],
                'val': ['/home/tqminh/AmodalSeg/data/std_data/COCOA/COCOA_annotations_detectron/COCO_amodal_val2014_with_classes.json',
                            '/home/tqminh/AmodalSeg/data/std_data/COCOA/annotations/instances_val2014.json']
            }

    for ann_type in ['train', 'val']:
        with open(ann_path[ann_type][0], "r") as f:
            data = json.load(f)
        
        data = proc_vis_mask_iseg(data)

        with open(ann_path[ann_type][1], "w") as f:
            json.dump(data, f)

        print('')
