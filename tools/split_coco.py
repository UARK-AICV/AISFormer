import os 
import os.path as osp
import json
import shutil
from tqdm import tqdm

#fill in target data with meta data from data['categories]

target_data = { 
    'categories': [
            # {"supercategory": "electronic","id": 72,"name": "tv"}, 
            # {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'}
            {'supercategory': 'person', 'id': 1, 'name': 'person'}
        ],
	'images': [],
	'annotations': []
}
DATA_DIR = '../data/std_data/coco'
SAVE_DIR = '../data/std_data/coco_person'
# DATA_DIR = '../data/std_data/coco'
# SAVE_DIR = '../data/std_data/coco_tv_firehydrant'

os.makedirs(osp.join(SAVE_DIR, 'train2017'), exist_ok=True)
os.makedirs(osp.join(SAVE_DIR, 'val2017'), exist_ok=True)
os.makedirs(osp.join(SAVE_DIR, 'annotations'), exist_ok=True)

category_ids = []
for cat_obj in target_data['categories']:
    category_ids.append(cat_obj['id'])

split_type = 'train'
file = open(os.path.join(DATA_DIR, 'annotations/instances_train2017.json'))
data = json.load(file)

image_id_to_filename_association = {}

"""
creates dictionary of assocation between image id and image file name
to avoid costly lookup. As a note, indice = id.png = file_name isn't always
true (refer to data['images'][1000] for an example)
"""
for img in data['images']:
    # image_id_to_filename_association[img['id']] = img['file_name']
    image_id_to_filename_association[img['id']] = img

ann_cnt = 0
img_cnt = 0
file_name_lists = []
fail_imgs = []
already_imgs = {}
img_id_count = {}
n_appear = 0
for ann in tqdm(data['annotations']):
    if ann['category_id'] in category_ids:
        ann_cnt += 1
        img_obj = image_id_to_filename_association[ann['image_id']]
        file_name = img_obj['file_name']
        if file_name == 'COCO_train2014_000000048432.jpg':
            n_appear += 1
        # if already_imgs.get(file_name) is None:
        #     already_imgs[file_name] = 1
        if file_name not in file_name_lists:
            img_cnt += 1
            file_name_lists.append(file_name)
            # img_obj['id'] = img_cnt
            target_data['images'].append(img_obj)

            file_path_src = '{}/{}2017/'.format(DATA_DIR, split_type) + file_name
            if osp.isfile(file_path_src) == False:
                fail_imgs.append(file_name)
                continue
            file_path_dest = SAVE_DIR + '/{}2017/'.format(split_type)
            shutil.copy(file_path_src, file_path_dest)

        # ann['id'] = ann_cnt
        ann['image_id'] = img_obj['id']

        if img_id_count.get(img_obj['id']) is None:
            img_id_count[img_obj['id']] = {
                'annots': [ann['id']], 'n_annots': 1, 'fname': file_name
            }   
        else:
            img_id_count[img_obj['id']]['annots'].append(ann_cnt)
            img_id_count[img_obj['id']]['n_annots'] += 1

        target_data['annotations'].append(ann)

print(f'Fail imgs {len(fail_imgs)} : {fail_imgs}')
split_annotation_file_path = SAVE_DIR +'/annotations/instances_{}2017.json'.format(split_type)
# print(f'n_appear: {n_appear}')
# with open('data/coco-car/split_coco.json', 'w') as f:
#     json.dump(img_id_count, f, indent=2)

json_object = json.dumps(target_data)
with open(split_annotation_file_path, 'w') as outfile:
    outfile.write(json_object) 
