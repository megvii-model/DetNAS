mkdir -p datasets/coco
ln -s /path/to/coco/annotations datasets/coco/annotations
ln -s /path/to/coco/train2014 datasets/coco/train2014
ln -s /path/to/coco/val2014 datasets/coco/val2014

mkdir ImageNet-Pretrain-models/

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python3 setup.py build_ext install
cd -

pip3 install torch==1.3.0 torchvision
pip3 install ninja yacs cython matplotlib
python3 setup.py build develop

cd maskrcnn_benchmark/pytorch_distributed_syncbn
bash compile.sh
cd -
