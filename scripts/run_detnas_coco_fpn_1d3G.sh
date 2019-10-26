export NGPUS=8
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/e2e_faster_rcnn_DETNAS_COCO_FPN_1d3G_2x.yaml  OUTPUT_DIR models/DETNAS_COCO_FPN_1d3G_2x_nosubmean_SyncBatchNorm_tong_FPNSYNCBN_BOXHEADSYNCBN
