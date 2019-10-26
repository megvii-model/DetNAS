export NGPUS=8
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/retinanet/retinanet_DETNAS_COCO_FPN_300M_1x.yaml  OUTPUT_DIR models/DETNAS_COCO_Retinanet_300M_1x_nosubmean_SyncBatchNorm_tong_FPN256 
