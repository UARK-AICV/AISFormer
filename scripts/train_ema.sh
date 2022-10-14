python3 tools/train_net_ema.py --num-gpus 1 \
	--config-file ${config_path} 2>&1 | tee log/train_log_$ID.txt
