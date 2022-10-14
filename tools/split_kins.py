import json, os
import shutil

'''
[{'supercategory': 'Living Thing', 'id': 1, 'name': 'cyclist'}, 
{'supercategory': 'Living Thing', 'id': 2, 'name': 'pedestrian'}, {'supercategory': 'vehicles', 'id': 4, 'name': 'car'}, 
{'supercategory': 'vehicles', 'id': 5, 'name': 'tram'}, {'supercategory': 'vehicles', 'id': 6, 'name': 'truck'}, 
{'supercategory': 'vehicles', 'id': 7, 'name': 'van'}, {'supercategory': 'vehicles', 'id': 8, 'name': 'misc'}]
'''
#fill in target data with meta data from data['categories]

target_data = { 
	'categories': [{'supercategory': 'vehicles', 'id': 4, 'name': 'car'}],
	'images': [],
	'annotations': []
}

category_id = target_data['categories'][0]['id']

split_type = 'val'
file = open('../data/std_data/KINS/annotations/instances_{}2017.json'.format(split_type))
data = json.load(file)

try:
    os.mkdir('../data/std_data/KINS_' + str(category_id))
    os.mkdir('../data/std_data/KINS_' + str(category_id) + '/annotations')
except:
    pass

try:
    os.mkdir('../data/std_data/KINS_' + str(category_id) + '/{}2017'.format(split_type))
except:
    pass



image_id_to_filename_association = {}

"""
creates dictionary of assocation between image id and image file name
to avoid costly lookup. As a note, indice = id.png = file_name isn't always
true (refer to data['images'][1000] for an example)
"""
for img in data['images']:
    image_id_to_filename_association[img['id']] = img

file_name_lists = []
for ann in data['annotations']:
    if ann['category_id'] == category_id:
        img_obj = image_id_to_filename_association[ann['image_id']]
        file_name = img_obj['file_name']
        if file_name not in file_name_lists:
            file_name_lists.append(file_name)
            target_data['images'].append(img_obj)

            file_path_src = '../data/std_data/KINS/{}2017/'.format(split_type) + file_name
            file_path_dest = '../data/std_data/KINS_' + str(category_id) + '/{}2017/'.format(split_type)
            shutil.copy(file_path_src, file_path_dest)

        assert ann['image_id'] == img_obj['id']
        ann['image_id'] = img_obj['id']
        target_data['annotations'].append(ann) 

split_annotation_file_path = '../data/std_data/KINS_' + \
        str(category_id) + '/annotations/instances_{}2017.json'.format(split_type)


json_object = json.dumps(target_data)

with open(split_annotation_file_path, 'w') as outfile:
    outfile.write(json_object)
