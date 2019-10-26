import torch
import apex
import os
from IPython import embed
from torch import nn
import torch.nn.functional as F
import argparse
import numpy as np
import torch.distributed as dist
from syncbn import DistributedSyncBN
from test_case import TestCase


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--sync_bn", action="store_true")
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    assert num_gpus == 2, "unittest only for 2 gpus"

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    device = torch.device('cuda')
    model = DistributedSyncBN(3).to(device)
    nn.init.constant_(model.weight, 1)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    # check train
    np.random.seed(args.local_rank)
    x = torch.from_numpy(np.random.rand(2, 3, 2, 2)).float()
    x.to(device)
    y = model(x)
    z = F.relu(y)
    z = z.sum()
    z.backward()

    np.random.seed(0)
    x1 = np.random.rand(2, 3, 2, 2)
    np.random.seed(1)
    x2 = np.random.rand(2, 3, 2, 2)
    x = np.concatenate((x1, x2), axis=0).astype(float)

    xv = x.reshape(4, 3, -1)
    mean = np.mean(np.mean(xv, axis=0, keepdims=True), axis=2, keepdims=True)
    a, b, c = xv.shape
    var = (np.var(np.transpose(xv, [0, 2, 1]).reshape((a * c, b)),
                  axis=0, ddof=1).reshape((1, b, 1)))

    sd = np.sqrt(var+model.module.eps)

    y_expect = (xv - mean) / sd
    y_expect = y_expect.reshape(x.shape)

    test_case = TestCase()

    if dist.get_rank() == 0:
        test_case.assertTensorClose(y_expect[:2], y.detach().cpu().numpy(), max_err=5e-6)
    else:
        test_case.assertTensorClose(y_expect[2:], y.detach().cpu().numpy(), max_err=5e-6)


if __name__ == "__main__":
    main()