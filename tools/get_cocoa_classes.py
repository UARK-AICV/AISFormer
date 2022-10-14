import json, pickle
if __name__=='__main__':
    ann_path = '/home/tqminh/AmodalSeg/data/std_data/COCOA/annotations/instances_val2014.json'

    with open(ann_path, "r") as f:
        ann_data = json.load(f)

    cat_list = []
    for ann_cat in ann_data['categories']:
        cat_list.append(ann_cat['name'])

    print(cat_list)
    with open('/home/tqminh/AmodalSeg/data/std_data/COCOA/cocoa_cat_list', 'wb') as fp:
        pickle.dump(cat_list, fp)

