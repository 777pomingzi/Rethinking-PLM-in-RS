import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pprint as pp
import os
from datetime import date
import pickle
from tqdm import tqdm


class Ranker(nn.Module):
    def __init__(self, metrics_ks):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, scores, labels, candidates):
        labels = labels.squeeze()

        try:
            loss = self.ce(scores, labels).item()
        except:
            loss = 0.0
        
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values

        if candidates is not None:
            scores = scores.gather(1, candidates)
        
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )

        return res + [loss]


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():
        user = np.random.randint(0, usernum)+1
        
        while len(user_train[user]) <= 1: user = np.random.randint(0, usernum)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
            
        return (user, seq, pos, neg)

    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()



def evaluate(model, optimizer,data_loader, mode, args, accum_iter, logger_service):
    average_meter_set = AverageMeterSet()
    with torch.no_grad():
            tqdm_dataloader = tqdm(data_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                if len(batch) == 3:
                    seqs, candidates, labels = batch
                    candidates = candidates.to(args.device)
                else:
                    seqs, labels = batch
                    candidates = None
                predictions = model.predict(seqs)
                ranker = Ranker(args.metric_ks)
                res = ranker(predictions, labels, candidates)
                metrics = {}
                for i, k in enumerate(args.metric_ks):
                    metrics["NDCG@%d" % k] = res[2*i]
                    metrics["Recall@%d" % k] = res[2*i+1]
                metrics["MRR"] = res[-3]
                metrics["AUC"] = res[-2]

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in args.metric_ks] +\
                                      ['Recall@%d' % k for k in args.metric_ks]
                description = mode+':' + ','.join(s + ' {:.4f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                description = description.replace('@', '')
                tqdm_dataloader.set_description(description)

            log_data = {
                    'state_dict': (create_state_dict(model,optimizer)),
                    'accum_iter': accum_iter,
                }
            average_metrics = average_meter_set.averages()
            log_data.update(average_metrics)
            if mode == 'Eval':
                logger_service.log_val(log_data)
            else:
                logger_service.log_test(log_data)
            return average_metrics
            

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)



class AverageMeterSet(object):
    def __init__(self,meters=None):
        self.meters=meters if meters else{}
    
    def __getitem__(self,key):
        if key not in self.meters:
            meter=AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self,name,value,n=1):
        if name not in self.meters:
            self.meters[name]=AverageMeter()
        self.meters[name].update(value,n)

    def reset(self):
        for meter in self.meters:
            meter.reset()
    
    def values(self,format_string='{}'):
        return {format_string.format(name):meter.val for name,meter in self.meters.items()}

    def averages(self,format_string='{}'):
        return {format_string.format(name):meter.avg for name,meter in self.meters.items()}

    def sums(self,format_string='{}'):
        return {format_string.format(name):meter.sum for name,meter in self.meters.items()}

    def counts(self,format_string='{}'):
        return {format_string.format(name):meter.count for name,meter in self.meters.items()}


class AverageMeter(object):

    def __init__(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    
    def update(self,val,n=1):
        self.val=val
        self.sum+=val
        self.count+=n
        self.avg=self.sum/self.count

    def __format__(self,format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self,format=format)


def setup_train(args):
    export_root=create_experiment_export_folder(args)
    if args.local_rank==0:
        pp.pprint({k:v for k,v in vars(args).items() if v is not None},width=1)
    return export_root

def create_experiment_export_folder(args):
    experiment_dir,experiment_description=args.experiment_dir,args.experiment_description
    if not os.path.exists(experiment_dir) and args.local_rank==0 :
        os.makedirs(experiment_dir)
    experiment_path=get_name_of_experiment_path(experiment_dir,experiment_description)
    if args.local_rank==0 and (not os.path.exists(experiment_path)):
        # print(args.local_rank)
        os.makedirs(experiment_path)
        print('Folder created: '+os.path.abspath(experiment_path))
    return experiment_path


def get_experiment_index(experiment_path):
    idx = 0 
    while os.path.exists(experiment_path+"_"+str(idx)):
        idx+=1
    return idx

    
def get_name_of_experiment_path(experiment_dir,experiment_description):
    experiment_path=os.path.join(experiment_dir,(experiment_description+"_"+str(date.today())))
    idx=get_experiment_index(experiment_path)
    experiment_path=experiment_path+"_"+str(idx)
    return experiment_path


def create_state_dict(model,optimizer):
        return {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

class AverageMeterSet(object):
    def __init__(self,meters=None):
        self.meters=meters if meters else{}
    
    def __getitem__(self,key):
        if key not in self.meters:
            meter=AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self,name,value,n=1):
        if name not in self.meters:
            self.meters[name]=AverageMeter()
        self.meters[name].update(value,n)

    def reset(self):
        for meter in self.meters:
            meter.reset()
    
    def values(self,format_string='{}'):
        return {format_string.format(name):meter.val for name,meter in self.meters.items()}

    def averages(self,format_string='{}'):
        return {format_string.format(name):meter.avg for name,meter in self.meters.items()}

    def sums(self,format_string='{}'):
        return {format_string.format(name):meter.sum for name,meter in self.meters.items()}

    def counts(self,format_string='{}'):
        return {format_string.format(name):meter.count for name,meter in self.meters.items()}


class AverageMeter(object):

    def __init__(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    
    def update(self,val,n=1):
        self.val=val
        self.sum+=val
        self.count+=n
        self.avg=self.sum/self.count

        
    def __format__(self,format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self,format=format)
