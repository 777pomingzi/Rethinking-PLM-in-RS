from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils
import random 
from torch.utils.data import DistributedSampler
from collections import Counter

class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        train_sampler = DistributedSampler(dataset)
        dataloader = data_utils.DataLoader(dataset,batch_size=self.args.train_batch_size, pin_memory=True,sampler=train_sampler)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        eval_sampler=DistributedSampler(dataset)
        dataloader = data_utils.DataLoader(dataset,batch_size=batch_size, pin_memory=True,sampler=eval_sampler)
        return dataloader

    def _get_eval_dataset(self, mode): 
        if mode == 'val':
            dataset = BertEvalDataset(self.train, self.val, None, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        else:
            dataset = BertEvalDataset(self.train, self.val, self.test, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = []
        labels = []
        ft_prob=self.rng.random()
        if ft_prob<0.91:
            for s in seq:
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(self.rng.randint(1, self.num_items))
                    else:
                        tokens.append(s)

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)
        else:
            tokens=seq[:-1]
            labels=[0]*len(tokens)+seq[-1:]
            tokens.append(self.mask_token)
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]



class BertEvalDataset(data_utils.Dataset):
    def __init__(self, train, val, test, max_len, mask_token, negative_samples):
        self.train =train
        self.val = val
        self.test = test

        self.users = sorted(self.val.keys())
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples
      

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]

        seq = self.train[user] if self.test is None else self.train[user]+self.val[user]
        label = self.val[user] if self.test is None else self.test[user]

        if self.negative_samples is not None:
            negs = self.negative_samples[user]
            candidates = label + negs
        

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        if self.negative_samples is not None:
            return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(label)
        else:
            return torch.LongTensor(seq), torch.LongTensor(label)
                

