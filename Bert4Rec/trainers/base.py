from itertools import accumulate
from loggers import *
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from utils import AverageMeterSet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from abc import *
from pathlib import Path

class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.args = args
        self.device = args.device
        if args.init_type is not None:
            embedding_dir = Path('../embedding')
            tuned_embedding_path = embedding_dir / self.args.dataset_code / self.args.init_type / ('item_embeddings_' + self.args.dataset_code)
            embeddings = torch.load(tuned_embedding_path)
            model.init_item_embedding(embeddings)

        self.model = model.to(self.device)

        self.is_parallel = args.num_gpu > 1

        print(torch.cuda.device_count)
        if self.is_parallel:
            self.model = nn.parallel.DistributedDataParallel(self.model,device_ids=[args.local_rank],
                                                      output_device=args.local_rank)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers,self.test_loggers = self._create_loggers()
        self.add_extra_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers,self.test_loggers)
        self.log_period_as_iter = args.log_period_as_iter

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    def train(self):
        accum_iter = 0
        best_target = float('-inf')
        patient = 10
        model_save_path = self.val_loggers[-1].get_save_path()
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            eval_metric = self.validate(epoch, accum_iter)
            if eval_metric['NDCG@10'] > best_target:
                best_target = eval_metric['NDCG@10']
                patient = 10
            else:
                patient -= 1
                if patient == 0:
                    break
        self.model.load_state_dict(torch.load(model_save_path, map_location=torch.device(self.args.device))['model_state_dict'])
        self.test(accum_iter)
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        self.lr_scheduler.step()
        self.train_loader.sampler.set_epoch(epoch)
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]
            # for gradient accumulation
            accumulation_steps=256/batch_size
            loss = self.calculate_loss(batch)/accumulation_steps
            loss.backward()
            
            if ((batch_idx+1)%accumulation_steps)==0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.4f} '.format(epoch+1, average_meter_set['loss'].avg))
            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()
        print('---------------Eval-----------------')
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:]]
                description = 'Val:' + ','.join(s + ' {:.4f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.replace('@', '')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_val(log_data)

            return  average_meter_set.averages()

    def test(self,accum_iter):
        self.model.eval()
        print('---------------Test-----------------')
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:]]
                description = 'Test:' + ','.join(s + ' {:.4f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.replace('@', '')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)
            log_data = {
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_test(log_data)
            
    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def _create_loggers(self):
        root = Path(self.export_root)
        
        writer = SummaryWriter(root.joinpath('logs_rank'+str(self.args.local_rank)))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]
        
        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint,self.args))
        val_loggers.append(BestModelLogger(model_checkpoint,self.args, metric_key=self.best_metric))

        test_loggers = []
        for k in self.metric_ks:
            test_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='test'))
            test_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='test'))

        return writer, train_loggers, val_loggers,test_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
