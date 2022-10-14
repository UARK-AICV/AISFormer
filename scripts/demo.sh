# python3 setup.py build develop #--no-deps
export PYTHONPATH=$PYTHONPATH:`pwd`
#export CUDA_LAUNCH_BLOCKING=1 # for debug
export CUDA_VISIBLE_DEVICES=0

method="aisformer"
model_dir="/home/tqminh/AmodalSeg/data/train_outputs/${method}/${method}_R_50_FPN_1x_amodal_kins"
python3 demo/demo.py --config-file ${model_dir}/config.yaml \
  --input /path/to/image.jpg \
  --output /path/to/outputfolder/ \
  --confidence-threshold 0.7 \
  --opts MODEL.WEIGHTS ${model_dir}/model_final.pth \

