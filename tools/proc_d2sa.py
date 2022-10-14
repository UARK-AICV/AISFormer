import os, json
from os.path import join


def proc_vis_mask_iseg(data_json):
    for ann in data_json['annotations']:
        ann['inmodal_seg'] = ann.pop('visible_mask')

    return data_json


if __name__=='__main__':
    image_path = '/home/tqminh/AmodalSeg/data/std_data/D2SA/d2s_amodal_images_v1/images'

    ann_path = {
                'train': ['/home/tqminh/AmodalSeg/data/std_data/D2SA/d2s_amodal_annotations_v1/D2S_amodal_training_rot0.json',
                            '/home/tqminh/AmodalSeg/data/std_data/D2SA/annotations/instances_train2017.json'],
                'val': ['/home/tqminh/AmodalSeg/data/std_data/D2SA/d2s_amodal_annotations_v1/D2S_amodal_validation.json',
                            '/home/tqminh/AmodalSeg/data/std_data/D2SA/annotations/instances_val2017.json'],
                'train_aug': ['/home/tqminh/AmodalSeg/data/std_data/D2SA/d2s_amodal_annotations_v1/D2S_amodal_augmented.json',
                            '/home/tqminh/AmodalSeg/data/std_data/D2SA/annotations/instances_train_aug_2017.json'],
            }

    for ann_type in ['train', 'val', 'train_aug']:
        with open(ann_path[ann_type][0], "r") as f:
            data = json.load(f)
        
        data = proc_vis_mask_iseg(data)

        with open(ann_path[ann_type][1], "w") as f:
            json.dump(data, f)

        print('')
