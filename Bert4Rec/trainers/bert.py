from .base import AbstractTrainer
import torch
import torch.nn as nn



class  Ranker(nn.Module):
    def __init__(self, metrics_ks):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, scores, labels, candidates):
        labels = labels.squeeze()
        try:
            loss = self.ce(scores, labels).item()
        except:
            loss = 0.0
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values
        if candidates is not None:
            scores = scores.gather(1,candidates)
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )
        return res + [loss]



class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)  # B x T x V
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        if len(batch) == 3:
            seqs, candidates, labels = batch
        else:
            seqs, labels = batch
            candidates = None
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        ranker = Ranker(self.metric_ks)
        res = ranker(scores, labels, candidates)
        metrics = {}
        
        for i, k in enumerate(self.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]                    
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]

        return metrics
