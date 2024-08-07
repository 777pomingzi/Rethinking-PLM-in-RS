import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
import pickle
from pytorch_lightning import seed_everything

from utils import read_json, AverageMeterSet, Ranker
from optimization import create_optimizer_and_scheduler
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerTrainDataset, RecformerEvalDataset

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def convert_save_embddings(embedding, save_path):

    pad= torch.zeros((1,768))
    new_emb=torch.cat((pad,embedding.detach().cpu()),dim=0)
    torch.save(new_emb,save_path)
    

def load_data(args):

    train = read_json(os.path.join(args.data_path, args.train_file), True)
    val = read_json(os.path.join(args.data_path, args.dev_file), True)
    test = read_json(os.path.join(args.data_path, args.test_file), True)
    item_meta_dict = json.load(open(os.path.join(args.data_path, args.meta_file)))
    
    item2id = read_json(os.path.join(args.data_path, args.item2id_file))
    id2item = {v:k for k, v in item2id.items()}

    item_meta_dict_filted = dict()
    for k, v in item_meta_dict.items():
        if k in item2id:
            item_meta_dict_filted[k] = v

    return train, val, test, item_meta_dict_filted, item2id, id2item


tokenizer_glb: RecformerTokenizer = None
def _par_tokenize_doc(doc):
    
    item_id, item_attr = doc

    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids, token_type_ids

def encode_all_items(model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args):

    model.eval()

    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]

    item_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(items), args.batch_size), ncols=100, desc='Encode all items'):

            item_batch = [[item] for item in items[i:i+args.batch_size]]

            inputs = tokenizer.batch_encode(item_batch, encode_item=False)

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v).to(args.device)

            outputs = model(**inputs)

            item_embeddings.append(outputs.pooler_output.detach())

    item_embeddings = torch.cat(item_embeddings, dim=0)#.cpu()

    return item_embeddings


def eval(model, dataloader, args):

    model.eval()

    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()

    for batch, labels, candidates in tqdm(dataloader, ncols=100, desc='Evaluate'):
        
        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = labels.to(args.device)
        labels -= 1
        if candidates[0] is not None:
            candidates = candidates.to(args.device)
            candidates -= 1

        with torch.no_grad():
            scores = model(**batch)     

        res = ranker(scores, labels, candidates)

        metrics = {}
        for i, k in enumerate(args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]

        for k, v in metrics.items():
            average_meter_set.update(k, v)

    average_metrics = average_meter_set.averages()

    for key in average_metrics:
        average_metrics[key]= round(average_metrics[key],4)
    return average_metrics


        

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, args):

    model.train()

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc='Training')):
        for k, v in batch.items():
            batch[k] = v.to(args.device)

        if args.fp16:
            with autocast():
                loss = model(**batch)
        else:
            loss = model(**batch)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer_was_run = scale_before <= scale_after
                optimizer.zero_grad()

                if optimizer_was_run:
                    scheduler.step()
            else:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                optimizer.zero_grad()


def negative_sampler_factory(train,val,test,item_count,seed,root_path,sample_size):
    if sample_size < 0:
        return None
    root_path=Path(root_path)
    save_path= root_path.joinpath('sample_size{}-seed{}.pkl'.format(sample_size, seed))

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


def main():
    parser = ArgumentParser()
    # path and file
    parser.add_argument('--longformer_ckpt', type=str, default=None, required=True)
    parser.add_argument('--pretrain_ckpt', type=str, default=None, required=True)
    parser.add_argument('--data_path', type=str, default=None, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--ckptf1', type=str, default='best_model_1.bin')
    parser.add_argument('--ckpt', type=str, default='best_model.bin')
    parser.add_argument('--model_name_or_path', type=str, default='longformer-base-4096')
    parser.add_argument('--train_file', type=str, default='train.json')
    parser.add_argument('--dev_file', type=str, default='val.json')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--item2id_file', type=str, default='smap.json')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')

    # data process
    parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
    parser.add_argument('--dataloader_num_workers', type=int, default=0)

    # model
    parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")

    # train
    parser.add_argument('--num_train_epochs', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--finetune_negative_sample_size', type=int, default=1000)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[1,5,10,20,50], help='ks for Metric@k')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--test_only', default=False,type=str2bool)
    

    args = parser.parse_args()
    print(args)
    seed_everything(42)
    args.device = torch.device('cuda:{}'.format(args.device)) if args.device>=0 else torch.device('cpu')

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = len(item2id)
    config.finetune_negative_sample_size = args.finetune_negative_sample_size
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    
    global tokenizer_glb
    tokenizer_glb = tokenizer

    path_corpus = Path(args.data_path)
    dir_preprocess = path_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_output = Path(args.output_dir) / path_corpus.name
    path_output.mkdir(exist_ok=True, parents=True)
    path_ckpt = path_output / args.ckpt

    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'

    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')
        pool = Pool(processes=args.preprocessing_num_workers)
        pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_meta_dict.items())
        doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        tokenized_items = {item2id[item_id]: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()

        torch.save(tokenized_items, path_tokenized_items)

    tokenized_items = torch.load(path_tokenized_items)
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    train_data = RecformerTrainDataset(train, collator=finetune_data_collator)

    neg_smaples = negative_sampler_factory(train,val,test,len(item2id),98765,args.data_path,args.sample_size)

    val_data = RecformerEvalDataset(train, val, test, neg_smaples, mode='val', collator=eval_data_collator)
    test_data = RecformerEvalDataset(train, val, test, neg_smaples ,mode='test', collator=eval_data_collator)

    
    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=train_data.collate_fn)
    dev_loader = DataLoader(val_data, 
                            batch_size=args.batch_size, 
                            collate_fn=val_data.collate_fn)
    test_loader = DataLoader(test_data, 
                            batch_size=args.batch_size, 
                            collate_fn=test_data.collate_fn)

    model = RecformerForSeqRec(config)

    embedding_dir = Path('../embedding')
    domain = args.data_path.split('/')[-1]
    dir_domain = embedding_dir / domain
    os.makedirs(dir_domain,exist_ok=True)

    longformer_dir = dir_domain / 'longformer'
    longformer_emb_path =  longformer_dir / ('item_embeddings_' + domain)
    if not longformer_emb_path.exists():
        longformer_dir.mkdir(exist_ok=True)
        longformer_ckpt = torch.load(args.longformer_ckpt)
        model.load_state_dict(longformer_ckpt, strict=False)
        model.to(args.device)
        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        convert_save_embddings(item_embeddings,longformer_emb_path)



    pretrain_ckpt = torch.load(args.pretrain_ckpt)
    model.load_state_dict(pretrain_ckpt, strict=False)

    model.to(args.device) # send item embeddings to device

    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    path_item_embeddings = dir_preprocess / f'item_embeddings_{path_corpus.name}'
    if path_item_embeddings.exists():
        print(f'[Item Embeddings] Use cache: {path_tokenized_items}')
    else:
        print(f'Encoding items.')
        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        torch.save(item_embeddings, path_item_embeddings)
    
    
    
    item_embeddings = torch.load(path_item_embeddings)
    model.init_item_embedding(item_embeddings)

    pretrain_dir = dir_domain / 'pretrain'
    pretrain_emb_path =  pretrain_dir / ('item_embeddings_' + domain)
    if not pretrain_emb_path.exists():
        pretrain_dir.mkdir(exist_ok=True)
        convert_save_embddings(item_embeddings,pretrain_emb_path)
        

    num_train_optimization_steps = int(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)
    
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")
   
    if args.test_only:
        test_metrics = eval(model, test_loader, args)
        print(f'Test set: {test_metrics}')
        return
    
    # stage1 finetune
    best_target = float('-inf')
    patient = 5
    

    for epoch in range(args.num_train_epochs):

        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        
        model.init_item_embedding(item_embeddings)

        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args)
        
        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args)
            print(f'Epoch: {epoch}. Dev set: {dev_metrics}')

            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 5
                torch.save(model.state_dict(), path_ckpt)
                torch.save(model.state_dict(), path_output / args.ckptf1)
            else:
                patient -= 1
                if patient == 0:
                    break
    
    print('Load best model in stage 1.')
    model.load_state_dict(torch.load(path_ckpt))
   
    finetune_dir = dir_domain / 'finetune'
    finetune_emb_path = finetune_dir / ('item_embeddings_' + domain)
    if not finetune_emb_path.exists():
        finetune_dir.mkdir(exist_ok=True)
        convert_save_embddings(model.item_embedding.weight,finetune_emb_path)
    patient = 5

    for epoch in range(args.num_train_epochs):

        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args)
        
        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args)
            print(f'Epoch: {epoch}. Dev set: {dev_metrics}')

            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 5
                torch.save(model.state_dict(), path_ckpt)
            
            else:
                patient -= 1
                if patient == 0:
                    break

    print('Test with the best checkpoint.')  
    model.load_state_dict(torch.load(path_ckpt))
    test_metrics = eval(model, test_loader, args)
    print(f'Test set: {test_metrics}')
               
if __name__ == "__main__":
    main()
