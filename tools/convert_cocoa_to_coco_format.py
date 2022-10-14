import json

cocoa_ann_path = "/home/tqminh/AmodalSeg/data/std_data/COCOA/annotations/COCO_amodal_train2014.json"

cocoa_ann = None

with open(cocoa_ann_path) as f:
    cocoa_ann = json.load(f)

import pdb;pdb.set_trace()
print("stuff")
