import torch.nn as nn
import torch
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding
class BERTEmbedding(nn.Module):
    def __init__(self,vocab_size,embed_size,max_len,drop_out=0.1):
        super().__init__()
        self.token=TokenEmbedding(vocab_size=vocab_size,embed_size=embed_size)
        self.position=PositionalEmbedding(max_len=max_len,d_model=embed_size)
        self.dropout=nn.Dropout(p=drop_out)
        self.embed_size=embed_size
    
    def init_item_embedding(self,embeddings):
        embeddings = torch.cat((embeddings,embeddings[:1]),dim=0)
        self.token = nn.Embedding.from_pretrained(embeddings,padding_idx=0)
        print('Initalize item embeddings from vectors.')
        
    def forward(self,sequence):
        x=self.token(sequence)+self.position(sequence)   #+ self.segment(segment_label)
        return self.dropout(x)