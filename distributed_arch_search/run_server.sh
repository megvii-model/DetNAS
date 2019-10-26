NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
export NCCL_IB_HCA
python3 distributed_arch_search/test_server.py --config-file configs/e2e_faster_rcnn_DETNAS_COCO_FPN_300M_search.yaml MODEL.WEIGHT models/DETNAS_COCO_FPN_300M_1x_search/model_final.pth
