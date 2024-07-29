from turtle import forward
import torch.nn as nn
from .gelu import GELU

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d__ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1=nn.Linear(d_model,d__ff)
        self.w_2=nn.Linear(d__ff,d_model)
        self.dropout=nn.Dropout(dropout)
        self.activation=GELU()

    def forward(self,x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))