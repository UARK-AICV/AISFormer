import json, os
from os.path import join
import shutil

annFile = "/home/tqminh/AmodalSeg/data/std_data/COCOA/annotations/instances_val2014.json"
srcImgFold = "/home/tqminh/AmodalSeg/data/std_data/COCOA/val2014"
dstImgFold = "/home/tqminh/AmodalSeg/data/std_data/COCOA/val_imgs"

try:
    os.mkdir(dstImgFold)
except:
    pass


with open(annFile, "r") as f:
    ann = json.load(f)

for img in ann['images']:
    print(img['file_name'])
    imgPath = join(srcImgFold, img['file_name'])
    dstPath = join(dstImgFold, img['file_name'])
    shutil.copy(imgPath, dstPath)



