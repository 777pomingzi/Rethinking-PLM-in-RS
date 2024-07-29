from torch.utils.data import Dataset
import torch


class SAS_Dataset(Dataset):
    def __init__(self,args,train,val,test,negative_samples):
        
        self.train = train
        self.val = val
        self.test = test
        self.max_len=args.maxlen
        self.users=list(self.val.keys())
        self.user_count=len(self.users)
        self.item_num=args.item_num
        self.negative_samples = negative_samples


    def __len__(self):
        return len(self.val)


    def __getitem__(self,index):
        user=self.users[index]

        seq = self.train[user] if self.test is None else self.train[user]+self.val[user]
        label = self.val[user] if self.test is None else self.test[user]

        if self.negative_samples is not None:
            negs=self.negative_samples[user]
            candidates = label + negs
        
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        seq = seq[-self.max_len:]

      
        if self.negative_samples is not None:
            return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(label)
        else:
            return torch.LongTensor(seq), torch.LongTensor(label)
                
        