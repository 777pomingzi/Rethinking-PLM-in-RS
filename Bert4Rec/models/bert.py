from .base import BaseModel
from .bert_modules.bert import BERT

import torch.nn as nn


class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.bert_num_items + 1)

    @classmethod
    def code(cls):
        return 'bert'
    
    def init_item_embedding(self,embeddings):
        self.bert.init_item_embedding(embeddings)

    def forward(self, x):
        x = self.bert(x)
        """
        item_embs = self.out.weight.unsqueeze(0)   #(1,I,H)
        logits = item_embs.matmul(x.unsqueeze(-1)).squeeze(-1)  #(B,H,1)
        return logits
        """
        return self.out(x)