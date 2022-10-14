#python3 setup.py build develop #--no-deps # for building d2
export PYTHONPATH=$PYTHONPATH:`pwd`
# export CUDA_LAUNCH_BLOCKING=1 # for debug
export CUDA_VISIBLE_DEVICES=2

ID=159


# trained model path and config
output_dir="../data/train_outputs/aisformer/aisformer_R_50_FPN_1x_amodal_kins"

python3 tools/train_net.py --num-gpus 1 \
        --config-file ${output_dir}/config.yaml \
        --eval-only MODEL.WEIGHTS ${output_dir}/model_final.pth 2>&1 | tee ${output_dir}/test_log.txt
