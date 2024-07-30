import os
from abc import ABCMeta, abstractmethod
import time
import torch
import argparse
import pickle
import pathlib as Path
from model import SASRec
from utils import *
from Dataset import SASdataset
from torch.utils.data import DataLoader,DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def needs_to_log(accum_iter,args):
        return accum_iter % args.log_period_as_iter < args.batch_size and accum_iter != 0

class AbstractBaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        pass

class MetricGraphPrinter(AbstractBaseLogger):
    def __init__(self, writer, key='train_loss', graph_name='Train Loss', group_name='metric'):
        self.key = key
        self.graph_label = graph_name
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, kwargs[self.key], kwargs['accum_iter'])
        else:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, 0, kwargs['accum_iter'])

    def complete(self, *args, **kwargs):
        self.writer.close()

class LoggerService(object):
    def __init__(self, train_loggers=None, val_loggers=None,test_loggers=None):
        self.train_loggers = train_loggers if train_loggers else []
        self.val_loggers = val_loggers if val_loggers else []
        self.test_loggers= test_loggers if test_loggers else []

    def complete(self, log_data):
        for logger in self.train_loggers:
            logger.complete(**log_data)
        for logger in self.val_loggers:
            logger.complete(**log_data)
        for logger in self.test_loggers:
            logger.complete(**log_data)    

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

    def log_val(self, log_data):
        for logger in self.val_loggers:
            logger.log(**log_data)

    def log_test(self, log_data):
        for logger in self.test_loggers:
            logger.log(**log_data)


def create_loggers(export_root,args):
        print(export_root)
        root = Path(export_root)
        writer = SummaryWriter(root)
        model_checkpoint = root.joinpath('models')
        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]
        val_loggers = []
        for k in args.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(BestModelLogger(args,model_checkpoint, metric_key=args.best_metric))
        test_loggers = []
        for k in args.metric_ks:
            test_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='test'))
            test_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='test'))
        
        return writer, train_loggers,val_loggers,test_loggers


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


class BestModelLogger(AbstractBaseLogger):
    def __init__(self,args, checkpoint_path, metric_key, filename='best_acc_model.pth'):
        self.checkpoint_path = checkpoint_path
        if args.local_rank==0 and not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.best_metric = 0.
        self.metric_key = metric_key
        self.filename = filename

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:
            self.best_metric = current_metric
            save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename)

    def get_save_path(self):
        return self.checkpoint_path.joinpath(self.filename)


def negative_sampler_factory(train,val,test,item_count,seed,root_path,sample_size):
    if sample_size < 0:
        return None
    root_path=Path(root_path)
    save_path= root_path.joinpath('random-sample_size{}-seed{}.pkl'.format(sample_size, seed))

    if save_path.is_file():
        print('Negatives samples exist. Loading.')
        negative_samples = pickle.load(save_path.open('rb'))
        return negative_samples

    print("Negative samples don't exist. Generating.")
    np.random.seed(seed)
    negative_samples = {}
    print('Sampling negative items')
    for user in tqdm(val):
        seen = set(train[user])
        seen.update(val[user])
        seen.update(test[user])
        samples = []
        for _ in range(sample_size):
            item = np.random.choice(item_count) + 1
            while item in seen or item in samples:
                item = np.random.choice(item_count) + 1
            samples.append(item)
        negative_samples[user] = samples

    with save_path.open('wb') as f:
        pickle.dump(negative_samples, f)

        return negative_samples



def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, type=str,required=True)
parser.add_argument('--min_uc', default=4, type=int)
parser.add_argument('--min_sc', default=0, type=int)
parser.add_argument('--rating_score', default=0, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=768, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--device_id', default=0, type=int)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1,5,10,20,50], help='Ks for Metric@k')
parser.add_argument("--local_rank", type=int, default=0,help="Number of cpu threads to use during batch generation")
parser.add_argument('--experiment_dir', type=str, default='Experiments')
parser.add_argument('--experiment_description', type=str, default='test')
parser.add_argument('--log_period_as_iter', type=int, default=12800)
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--sample_size', type=int, default=-1, help='Random sample size')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')
parser.add_argument('--init_type', type=str, default=None, help='The type of initialized embedding')
args = parser.parse_args()


if __name__ == '__main__':
    if args.device == 'cuda':
        args.device = 'cuda:' + str(args.device_id)
   
    fix_random_seed_as(args.seed)

    data_path = Path('preprocessed')
    folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-splitleave_one_out'\
    .format(args.dataset,args.rating_score,args.min_uc,args.min_sc)
    save_folder = data_path.joinpath(folder_name)
    data_path = save_folder.joinpath('dataset.pkl')
    dataset = load_pickle(data_path)

    
    user_train = dataset['train']
    user_val = dataset['val']
    user_test = dataset['test']

    num_batch = len(user_train) // args.batch_size
    usernum = len(user_train)
    itemnum = len(dataset['smap'])
    args.item_num=itemnum
    
    neg_samples = negative_sampler_factory(user_train,user_val,user_test,itemnum,98765,save_folder,args.sample_size)


    val_dataset = SASdataset.SAS_Dataset(args,user_train,user_val,None,neg_samples)
    test_dataset = SASdataset.SAS_Dataset(args,user_train,user_val,user_test,neg_samples)
    val_dataloader = DataLoader(val_dataset,batch_size=args.batch_size, pin_memory=True)
    test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size, pin_memory=True)

    if args.local_rank==0:
        writer, train_loggers,val_loggers,test_loggers = create_loggers(setup_train(args),args)
        logger_service = LoggerService(train_loggers,val_loggers,test_loggers)
    
    model_save_path = val_loggers[-1].get_save_path()

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device))['model_state_dict'])
        except: 
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
    
    if args.sample_size > 0:
        loss_fct = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1) # torch.nn.BCELoss()
    
    T = 0.0
    t0 = time.time()
    accum_iter = 0
    best_target = float('-inf')
    patient = 10

    if args.inference_only:
        model.eval()
        test_metric = evaluate(model, test_dataloader,'Test', args,accum_iter,logger_service)
    if args.init_type is not None:
        embedding_dir = Path('../embedding')
        tuned_embedding_path = embedding_dir / args.dataset / args.init_type / 'item_embeddings'
        embeddings = torch.load(tuned_embedding_path)
        model.init_item_embedding(embeddings)

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        print("epoch: ", epoch, " ", end=None)
        average_meter_set = AverageMeterSet()
        for step in tqdm(range(num_batch)): 
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            if args.sample_size > 0:
                pos_logits, neg_logits = model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                indices = np.where(pos != 0)
                loss = loss_fct(pos_logits[indices], pos_labels[indices])
                loss += loss_fct(neg_logits[indices], neg_labels[indices])
            else:
                logits = model(u, seq, None, None)
                loss = loss_fct(logits.view(-1,logits.shape[-1]), torch.LongTensor(pos).view(-1).to(args.device)) 
            adam_optimizer.zero_grad()
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            accum_iter += args.batch_size
            average_meter_set.update('loss',loss.item())
            if needs_to_log(accum_iter,args) and args.local_rank==0:
                log_data = {
                    'state_dict': (create_state_dict(model,adam_optimizer)),
                    'epoch': epoch,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                logger_service.log_train(log_data)
                
        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            eval_metric = evaluate(model, adam_optimizer, val_dataloader,'Eval', args, accum_iter, logger_service)
            if eval_metric['NDCG@10'] > best_target:
                best_target = eval_metric['NDCG@10']
                patient = 10
            else:
                patient -= 1
                if patient == 0:
                    break
            t0 = time.time()
            model.train()
        
    sampler.close()
    
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device(args.device))['model_state_dict'])
    model.eval()
    test_metric = evaluate(model,adam_optimizer ,test_dataloader,'Test', args,accum_iter,logger_service)

    print("Done")





