import os
import sys
import time
import glob
import numpy as np
import pickle
import torch
import logging
import argparse
import functools


from maskrcnn_benchmark.modeling.detector.generalized_rcnn import GeneralizedRCNN
from maskrcnn_benchmark.utils.complexity import Complexity
from maskrcnn_benchmark.config import cfg
from arch_search_config import config
from test_server import TestClient

sys.setrecursionlimit(10000)

print=functools.partial(print,flush=True)
choice=lambda x:x[np.random.randint(len(x))] if isinstance(x,tuple) else choice(tuple(x))


class EvolutionTrainer(object):
    def __init__(self,cfg,*,refresh=False):

        self.log_dir=cfg.OUTPUT_DIR
        self.checkpoint_name=os.path.join(self.log_dir,'checkpoint.brainpkl')

        self.refresh=refresh

        self.tester = TestClient()
        self.tester.connect()

        self.model = GeneralizedRCNN(cfg)
        self.complexity_info=Complexity()

        self.memory=[]
        self.candidates=[]
        self.vis_dict={}
        self.keep_top_k = {config.select_num:[],50:[]}
        self.epoch=0

    def save_checkpoint(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        info={}
        info['memory']=self.memory
        info['candidates']=self.candidates
        info['vis_dict']=self.vis_dict
        info['keep_top_k']=self.keep_top_k
        info['epoch']=self.epoch
        info['tester']=self.tester.save()
        # io.dump(info,self.checkpoint_name)
        with open(self.checkpoint_name, 'wb') as fid:
            pickle.dump(info, fid, pickle.HIGHEST_PROTOCOL)
        print('save checkpoint to',self.checkpoint_name)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        with open(self.checkpoint_name, 'rb') as fid:
            info = pickle.load(fid)
        # info=io.load(self.checkpoint_name)
        self.memory=info['memory']
        self.candidates = info['candidates']
        self.vis_dict=info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']
        self.tester.load(info['tester'])

        if self.refresh:
            for i,j in self.vis_dict.items():
                for k in ['test_key','speed_key']:
                    if k in j:
                        j.pop(k)
            self.refresh=False

        print('load checkpoint from',self.checkpoint_name)
        return True

    def legal(self,cand):
        assert isinstance(cand,tuple) and len(cand)==len(config.states)
        if cand not in self.vis_dict:
            self.vis_dict[cand]={}
        info=self.vis_dict[cand]
        if 'visited' in info:
            return False

        if config.flops_limit is not None:
            if 'flops' not in info:
                net=self.model.backbone
                inp = torch.randn(1, 3, 224, 224)
                flops = self.complexity_info(net, inp, cand)
                info['flops']=flops
            flops=info['flops']
            print('flops:',flops)
            if flops>config.flops_limit:
                return False
            info['flops']=flops

        self.vis_dict[cand]=info

        return True

    def update_top_k(self,candidates,*,k,key,reverse=False):
        assert k in self.keep_top_k
        print('select ......')
        t=self.keep_top_k[k]
        t+=candidates
        t.sort(key=key,reverse=reverse)
        self.keep_top_k[k]=t[:k]

    def sync_candidates(self):
        while True:
            ok=True
            for cand in self.candidates:
                info=self.vis_dict[cand]
                if 'acc' in info:
                    continue
                ok=False
                if 'test_key' not in info:
                    info['test_key']=self.tester.send(cand)

            self.save_checkpoint()

            for cand in self.candidates:
                info=self.vis_dict[cand]
                if 'acc' in info:
                    continue
                key=info.pop('test_key')

                try:
                    print('try to get',key)
                    res=self.tester.get(key,timeout=1800)
                    print(res)
                    info['acc']=res['acc']
                    info['err']=1-info['acc']
                    self.save_checkpoint()
                except:
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)

            time.sleep(5)
            if ok:
                break

    def stack_random_cand(self,random_func,*,batchsize=10):
        while True:
            cands=[random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand]={}
                info=self.vis_dict[cand]

            for cand in cands:
                yield cand

    def random_can(self,num):
        print('random select ........')
        candidates = []
        cand_iter=self.stack_random_cand(
            lambda:tuple(np.random.randint(i) for i in config.states))
        while len(candidates)<num:
            cand=next(cand_iter)

            if not self.legal(cand):
                continue
            candidates.append(cand)
            print('random {}/{}'.format(len(candidates),num))

        print('random_num = {}'.format(len(candidates)))
        return candidates

    def get_mutation(self,k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num*10

        def random_func():
            cand=list(choice(self.keep_top_k[k]))
            for i in range(len(config.states)):
                if np.random.random_sample()<m_prob:
                    cand[i]=np.random.randint(config.states[i])
            return tuple(cand)

        cand_iter=self.stack_random_cand(random_func)
        while len(res)<mutation_num and max_iters>0:
            cand=next(cand_iter)
            if not self.legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res),mutation_num))
            max_iters-=1

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self,k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num
        def random_func():
            p1=choice(self.keep_top_k[k])
            p2=choice(self.keep_top_k[k])
            return tuple(choice([i,j]) for i,j in zip(p1,p2))
        cand_iter=self.stack_random_cand(random_func)
        while len(res)<crossover_num and max_iters>0:
            cand=next(cand_iter)
            if not self.legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res),crossover_num))
            max_iters-=1

        print('crossover_num = {}'.format(len(res)))
        return res

    def train(self):
        print('population_num = {} select_num = {} mutation_num = {} '
              'crossover_num = {} random_num = {} max_epochs = {}'.format(
                config.population_num, config.select_num, config.mutation_num,
                config.crossover_num,
                config.population_num - config.mutation_num - config.crossover_num,
                config.max_epochs))

        if not self.load_checkpoint():
            self.candidates = self.random_can(config.population_num)
            self.save_checkpoint()

        while self.epoch<config.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.sync_candidates()

            print('sync finish')

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)
                self.vis_dict[cand]['visited'] = True

            self.update_top_k(self.candidates,k=config.select_num,key=lambda x:self.vis_dict[x]['err'])
            self.update_top_k(self.candidates,k=50,key=lambda x:self.vis_dict[x]['err'] )

            print('epoch = {} : top {} result'.format(self.epoch, len(self.keep_top_k[50])))
            for i,cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 err = {}'.format(i+1, cand, self.vis_dict[cand]['err']))
                ops = [config.blocks_keys[i] for i in cand]
                print(ops)

            mutation = self.get_mutation(config.select_num, config.mutation_num, config.m_prob)
            crossover = self.get_crossover(config.select_num,config.crossover_num)
            rand = self.random_can(config.population_num - len(mutation) -len(crossover))

            self.candidates = mutation+crossover+rand

            self.epoch+=1
            self.save_checkpoint()

        print(self.keep_top_k[config.select_num])
        print('finish!')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-r','--refresh',action='store_true')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args=parser.parse_args()
    refresh=args.refresh

    t = time.time()

    cfg.merge_from_file(args.config_file)
    cfg.OUTPUT_DIR = config.log_dir
    cfg.freeze()
    trainer=EvolutionTrainer(cfg,refresh=refresh)

    trainer.train()
    print('total searching time = {:.2f} hours'.format((time.time()-t)/3600))


if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)

