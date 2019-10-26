export NGPUS=8
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file configs/e2e_faster_rcnn_DETNAS_COCO_FPN_300M_search.yaml  OUTPUT_DIR models/DETNAS_COCO_FPN_300M_1x_search
