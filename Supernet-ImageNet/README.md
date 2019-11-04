# Supernet training on ImageNet
This part is modified from the [ImageNet project](https://github.com/megvii-model/SinglePathOneShot) and would be merged into it in the future.

## Usage

### 1. Setup Dataset and Flops Table

Download the ImageNet Dataset and move validation images to labeled subfolders. To do this, you can use the following script: [https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

Download the flops table to accelerate Flops calculation which is required in Uniform Sampling. It can be found in `$Link/op_flops_dict.pkl`.

We recommend to create a folder `data` and use it in both Supernet training and Evaluation training.

Here is a example structure of `data`:

```
data
|--- train                 ImageNet Training Dataset
|--- val                   ImageNet Validation Dataset
|--- op_flops_dict.pkl     Flops Table
```

### 2. Train Supernet

Train supernet with the following command:

```bash
cd src/Supernet
python3 train.py --train-dir $YOUR_TRAINDATASET_PATH --val-dir $YOUR_VALDATASET_PATH --model-size DETNAS-300M
```
