from venv import create
from config import*

import json 
import os
import pprint as pp
import random 
from datetime import date
from pathlib import Path

import numpy as np
import torch 
import torch.backends.cudnn as cudnn
from torch import optim as optim

def setup_train(args):
    export_root=create_experiment_export_folder(args)
    export_experiments_config_as_json(args,export_root)

    pp.pprint({k:v for k,v in vars(args).items() if v is not None},width=1)
    return export_root

def create_experiment_export_folder(args):
    experiment_dir,experiment_description=args.experiment_dir,args.experiment_description
    if not os.path.exists(experiment_dir) and args.local_rank==0:
         os.makedirs(experiment_dir)
    experiment_path=get_name_of_experiment_path(experiment_dir,experiment_description)
    if args.local_rank==0:
        os.makedirs(experiment_path)
    print('Folder created: '+os.path.abspath(experiment_path))
    return experiment_path

def get_name_of_experiment_path(experiment_dir,experiment_description):
    experiment_path=os.path.join(experiment_dir,(experiment_description+"_"+str(date.today())))
    idx=get_experiment_index(experiment_path)
    experiment_path=experiment_path+"_"+str(idx)
    return experiment_path

def get_experiment_index(experiment_path):
    idx = 0 
    while os.path.exists(experiment_path+"_"+str(idx)):
        idx+=1
    return idx

def export_experiments_config_as_json(args,experiment_path):
    if args.local_rank==0:
        with open(os.path.join(experiment_path,'config.json'),'w') as outfile:
            json.dump(vars(args),outfile,indent=2)

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic=True
    cudnn.benchmark=False

def create_optimizer(model,args):
    if args.optimizer=='Adam':
        return optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    return optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)


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