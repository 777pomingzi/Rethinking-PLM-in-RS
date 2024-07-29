from re import template
import torch

def set_template(args):
    if args.template is None:
        return 
    elif args.template.startswith('train_bert'):
        args.mode='train'
        torch.distributed.init_process_group(backend="nccl",init_method='env://')
        args.local_rank=torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        args.dataset_code='Pantry'
        args.min_rating=0
        args.min_uc=4
        args.min_sc=0
        args.split='leave_one_out'
        args.dataloader_code='bert'
        batch=256
        args.train_batch_size=batch
        args.val_batch_size=batch
        args.test_batch_size=batch
        args.test_negative_sampler_code='random'
        args.test_negative_sample_size=100
        args.test_negative_sampling_seed=98765

        args.init_type = 'finetune'
        args.trainer_code='bert'
        args.device='cuda'
        args.num_gpu=torch.cuda.device_count()
        args.optimizer='Adam'
        args.lr=0.0001
        args.decay_step=25
        args.gamma=1.0
        args.num_epochs=200
        args.metric_ks=[1,5,10,20,50]
        args.best_metric='NDCG@10'

        args.model_code='bert'
        args.model_init_seed=0
        num_users,num_items=get_user_item_nums(args)

        args.bert_dropout=0.1
        args.bert_hidden_units=768
        args.bert_mask_prob=0.15
        args.bert_max_len=50
        args.bert_num_blocks=2
        args.bert_num_heads=4
        args.bert_num_items=num_items

def get_user_item_nums(args):
    if args.dataset_code == 'Movies':
        if args.min_rating ==0 and args.min_uc==5 and args.min_sc==5:
            return 26342,44410
        else:
            raise ValueError()
    if args.dataset_code == 'Sports':
        if args.min_rating ==0 and args.min_uc==5 and args.min_sc==5:
            return  21801,87235
        else:
            raise ValueError()
    if args.dataset_code =='Arts':
        if args.min_rating ==0 and args.min_uc==4 and args.min_sc==0:
            return 95095,138116
        else:
            raise ValueError()
    if args.dataset_code =='Music':
        if args.min_rating ==0 and args.min_uc==4 and args.min_sc==0:
            return 49592,53899
        else:
            raise ValueError()
    if args.dataset_code=='Pantry': 
        if args.min_rating ==0 and args.min_uc==4 and args.min_sc==0:
            return 21300,8249
        else:
            raise ValueError()
    if args.dataset_code=='Steam':
        if args.min_rating ==0 and args.min_uc==5 and args.min_sc==5:
            return 33118,6638
        else:
            raise ValueError()
    raise ValueError()