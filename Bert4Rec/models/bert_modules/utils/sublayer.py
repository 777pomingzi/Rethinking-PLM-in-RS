import torch.nn as nn
from .layer_norm import LayerNorm

class SublayerConnection(nn.Module):
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,sublayer):
        return self.norm(x+self.dropout(sublayer(x)))
        #return x+self.dropout(sublayer(self.norm(x)))