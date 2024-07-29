from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *

def train():
    print("---------------device---------------------")
    print(torch.cuda.device_count())
    export_root=setup_train(args)
    model=model_factory(args)
    train_loader,val_loader,test_loader=dataloader_factory(args)
    trainer=trainer_factory(args,model,train_loader,val_loader,test_loader,export_root)
    trainer.train()


if __name__=='__main__':
    if args.mode=='train':
        train()
    else:
        raise ValueError('Invalid mode')