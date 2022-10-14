import json
from collections import Counter

annPath = '/data/tqminh/AmodalSeg/std_data/COCOA/annotations/COCO_amodal_train2014.json'


with open(annPath, 'r') as f:
    data =json.load(f)

cats = []
for ann in data['annotations']:
    for reg in ann['regions']:
        cats.append(reg['name'])

print(Counter(cats))
print(len(set(cats)))
