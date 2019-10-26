#!/usr/bin/env python3
from multiprocessing import Process
from multiprocessing import Queue
import argparse
import logging
import pickle
import shutil

import os
import sys
import time
import hashlib
import glob
import re
import gc
import uuid
import numpy as np
from tqdm import tqdm
import tempfile

import functools
print=functools.partial(print,flush=True)


import torch
import torch.nn as nn
import torch.multiprocessing as mp
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import (prepare_for_coco_detection,
     evaluate_predictions_on_coco, _accumulate_predictions_from_multiple_gpus, COCOResults)
from maskrcnn_benchmark.modeling.detector.generalized_rcnn import GeneralizedRCNN
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.comm import is_main_process
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

from mq_server_base import MessageQueueServerBase,MessageQueueClientBase
from arch_search_config import config


class DetectronCheckpointer2(DetectronCheckpointer):
    """docstring for DetectronCheckpointer2"""
    def __init__(self, cfg, model, save_dir, save_to_disk):
        super(DetectronCheckpointer2, self).__init__(cfg, model, 
                save_dir=save_dir, save_to_disk=save_to_disk)

    def load(self, f=None):
        if not f:
            # no checkpoint could be found
            print("No checkpoint found. Initializing model from scratch")
            return {}
        print("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            print("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            print("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        # self.logger.info("Saving checkpoint to {}".format(save_file))
        print("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)


def compute_on_dataset(model, rngs, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in tqdm(enumerate(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images, rngs=rngs)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def inference(
    model,
    rngs,
    data_loader,
    iou_types=("bbox",),
    box_only=False,
    device="cuda",
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
):

    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    dataset = data_loader.dataset
    predictions = compute_on_dataset(model, rngs, data_loader, device)
    # wait for all processes to complete before measuring the time
    synchronize()

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    coco_results = {}
    if "bbox" in iou_types:
        coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset)

    results = COCOResults(*iou_types)
    uuid1 = str(uuid.uuid1())

    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(
                    output_folder, uuid1 + iou_type + ".json")
            res = evaluate_predictions_on_coco(
                dataset.coco, coco_results[iou_type], file_path, iou_type
            )
            results.update(res)

        if os.path.isfile(file_path):
            os.remove(file_path)

    return results


def fitness(gpu, ngpus_per_node, cfg, args, rngs, salt, conn):
    num_gpus = int(os.environ["WORLD_SIZE"]) \
        if "WORLD_SIZE" in os.environ else 1
    args["distributed"] = num_gpus > 1

    args["local_rank"] = gpu

    if args["distributed"]:
        torch.cuda.set_device(args["local_rank"])
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            world_size=num_gpus, rank=args["local_rank"]
        )

    model = GeneralizedRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer2(
        cfg, model, save_dir=cfg.OUTPUT_DIR, save_to_disk=save_to_disk)
    extra_checkpoint_data = checkpointer.load(os.path.join(cfg.OUTPUT_DIR, salt+".pth"))

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    if cfg.OUTPUT_DIR:
        dataset_names = cfg.DATASETS.TEST
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(
                cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    data_loaders_val = make_data_loader(
        cfg, is_train=False, is_distributed=args["distributed"])
    for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
        results = inference(
            model,
            rngs,
            data_loader_val,
            iou_types=iou_types,
            box_only=False,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()

    if get_rank() == 0:
        conn.send(results.results['bbox']['AP'])
        conn.close()


def bn_statistic(gpu, ngpus_per_node, cfg, args, rngs, conn):
    num_gpus = int(os.environ["WORLD_SIZE"]) \
        if "WORLD_SIZE" in os.environ else 1
    args["distributed"] = num_gpus > 1
    
    args["local_rank"] = gpu

    if args["distributed"]:
        torch.cuda.set_device(args["local_rank"])
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            world_size=num_gpus, rank=args["local_rank"]
        )

    model = GeneralizedRCNN(cfg)

    device = cfg.MODEL.DEVICE
    model.to(device)

    if args["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args["local_rank"]],
            output_device=args["local_rank"],
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer2(
        cfg, model, save_dir=output_dir, save_to_disk=save_to_disk)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    for name, param in model.module.named_buffers():
        if 'running_mean' in name:
            nn.init.constant_(param, 0)
        if 'running_var' in name:
            nn.init.constant_(param, 1)

    data_loader = make_data_loader(
        cfg,
        is_train=True, 
        is_distributed=args["distributed"]
    )

    model.train()

    pbar = tqdm(total=500)
    for iteration, (images, targets, _) in enumerate(data_loader, 1):
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            loss_dict = model(images, targets, rngs)
        pbar.update(1)
        if iteration >= 500:
            break

    pbar.close()

    salt = str(uuid.uuid1())
    checkpointer.save(salt)

    if get_rank() == 0:
        conn.send(salt)
        conn.close()


class TorchMonitor(object):
    def __init__(self):
        self.obj_set=set()
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj not in self.obj_set:
                self.obj_set.add(obj)
    def find_leak_tensor(self):
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj not in self.obj_set:
                print(obj.size())


class TestClient(MessageQueueClientBase):
    def __init__(self):
        super().__init__(config.host, config.port, config.username,
                            config.test_send_pipe, config.test_recv_pipe)
    def send(self,cand):
        assert isinstance(cand,tuple)
        return super().send(cand)


class TestServer(MessageQueueServerBase):
    def __init__(self, ngpus_per_node):
        super().__init__(config.host, config.port, config.username,
                            config.test_send_pipe, config.test_recv_pipe)
        self.ngpus_per_node=ngpus_per_node

    def eval(self, cand):
        res = self._test_candidate(cand)
        return res

    def _test_candidate(self, cand):
        res = dict()
        try:
            t0 = time.time()
            print('starting inference...')
            acc = self._inference(cand)
            print('time: {}s'.format(time.time() - t0))
            res = {'status': 'success', 'acc': acc}
            return res
        except:
            import traceback
            traceback.print_exc()
            res['status'] = 'failure'
            return res

    def _inference(self, cand):
        # bn_statistic
        parent_conn, child_conn = mp.Pipe()
        args = dict({"local_rank": 0, "distributed": False})
        mp.spawn(
            bn_statistic, nprocs=self.ngpus_per_node,
            args=(self.ngpus_per_node, cfg, args, cand, child_conn))
        salt = parent_conn.recv()

        # fitness
        parent_conn, child_conn = mp.Pipe()
        args = dict({"local_rank": 0, "distributed": False})
        mp.spawn(
            fitness, nprocs=self.ngpus_per_node,
            args=(self.ngpus_per_node, cfg, args, cand, salt, child_conn))

        if os.path.isfile(os.path.join(cfg.OUTPUT_DIR, salt+".pth")):
            os.remove(os.path.join(cfg.OUTPUT_DIR, salt+".pth"))

        return parent_conn.recv()


def main():
    parser=argparse.ArgumentParser()
    # parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-p', '--process', type=int, default=1)
    parser.add_argument('-r', '--reset', action='store_true')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")

    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")
    args=parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    ngpus_per_node = torch.cuda.device_count()
    os.environ["WORLD_SIZE"] = str(ngpus_per_node)
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)

    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.MAX_ITER = 88888888
    cfg.TEST.IMS_PER_BATCH = ngpus_per_node
    cfg.OUTPUT_DIR = config.log_dir
    cfg.freeze()

    train_server = TestServer(ngpus_per_node)
    train_server.run(args.process, reset_pipe=args.reset)


if __name__ == "__main__":
    try:
        main()
    except:
        import traceback
        traceback.print_exc()
        print(flush=True)
        os._exit(1)
