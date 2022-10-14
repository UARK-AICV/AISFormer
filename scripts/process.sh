python -m detectron2.data.datasets.process_data_amodal \
   /home/tqminh/AmodalSeg/data/std_data/D2SA/annotations/instances_train_aug_2017.json \
   /home/tqminh/AmodalSeg/data/std_data/D2SA/d2s_amodal_images_v1/images \
   d2sa_train_aug

python -m detectron2.data.datasets.process_data_inmodal \
   /home/tqminh/AmodalSeg/data/std_data/D2SA/annotations/instances_train2017.json \
   /home/tqminh/AmodalSeg/data/std_data/D2SA/d2s_amodal_images_v1/images \
   d2sa_train
