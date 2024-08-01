# Rethinking-PLM-in-RS

This repository contains the code of our RecSys 2024 paper **"[The Elephant in the Room: Rethinking the Usage of Pre-trained Language Model in Sequential Recommendation](https://arxiv.org/abs/2404.08796)"**.

## Quick Links

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Download Files](#Download-Files)
- [Files Preparation](#Files-Preparation)
- [Training](#Training)
- [Pre-training from scratch](#pre-training-from-scratch)
- [Acknowledgment](#Acknowledgment)

## Overview

In this paper, we identify significant model under-utilization and redundancy of PLM in behavior sequence modeling and propose to utilize the behavior-tuned PLMs initialized item embeddings to enhance the performance of conventional ID-based SR models. Firstly, we follow the training procedures of RECFORMER to obtain a behavior-tuned PLM. We then use this model to initialize the item embeddings of ID-based SR models with their corresponding textual representations and train these ID-based models as their original settings. Our repository provides code for obtaining initialized embeddings with RECFORMER and enhancing SASRec and BERT4Rec based on these embeddings.

## Dependencies

We train and test the model using the following main dependencies:

- Python 3.10.10
- PyTorch 2.0.0
- PyTorch Lightning 2.0.0 
- Transformers 4.35.2
- Deepspeed  0.12.4  

## Download Files

Please download the processed downstream (or pre-training, if needed) datasets, the pre-trained model, and embedding from Google Drive according to your needs:

| Model                  |
| :--------------------- |
| [Longformer_ckpt](https://drive.google.com/file/d/1NcCXRYQkSkVfDGaaIK3x_vQ6cAxjZQP0/view?usp=drive_link)    |
| [RecformerForSeqRec](https://drive.google.com/file/d/1zKABB_0_QNs3eaHQ6BPqwg_fWa6S_YXg/view?usp=drive_link) |

| Data                        |
| --------------------------- |
| [ID-based SR model data](https://drive.google.com/file/d/1NRoANHJsyKFthcIjkLH317WN-M7_WR6F/view?usp=drive_link)  |
| [RECFORMER_finetune_data](https://drive.google.com/file/d/1jWCrgBGeWXN7MFKNfR1WZWOxkU31Eq4H/view?usp=drive_link) |
| [RECFORMER_pretrain_data](https://drive.google.com/file/d/1sX04nmryDtHaNCv_Qob2pPGObttELyyV/view?usp=drive_link) |

| Embedding                     |
| ----------------------------- |
| [PLM-initialized embedding](https://drive.google.com/file/d/1NLTzfd148lX04_or2uLiC5VF57LEuh7V/view?usp=drive_link) |

If you want to replicate the pre-training and fine-tuning procedures of RECFORMER and use it to generate embeddings for initialization, you may consider downloading the Longformer_ckpt, RecformerForSeqRec, RECFORMER_pretrain_data, and RECFORMER_finetune_data.

If you want to directly replicate our results in SASRec and BERT4Rec using initialized embeddings, you may choose to download the ID-based SR model data and PLM-initialized embedding.

## Files Preparation

After downloading the files according to your needs, you could organize them as follows:

```
├── Bert4Rec
│   └── Data 
│        └── preprocessed (ID-based SR model data)
├── embedding (PLM-initialized-embedding)
├── Recformer
│   └── finetune_data (RECFORMER_finetune_data)
│   └── longformer_ckpt (Longformer_ckpt)
│   └── pretrain_ckpt (RecformerForSeqRec)
│   └── pretrain_data (RECFORMER_pretrain_data)
├── SASRec
│   └── preprocessed (ID-based SR model data)
```

## Training

Firstly, we train  `RecformerForSeqRec` with two-stage finetuning like its original paper and let it generate the item embeddings for initializing the ID-based SR models. A script is provided for finetuning in the Recformer folder:

```bash
cd Recformer
bash finetune.sh
```

You need to set the processed data path `--data_path`, evaluation setting `--sample_size` (adopting random sampling setting if sample_size is set to be greater than 0, adopting full-ranking setting if sample_size is set to be lower than 0).

Then, we could train the enhanced SASRec or BERT4Rec with initialized item embeddings, scripts are provided in their corresponding folders:

```bash
cd SASRec
bash finetune.sh
```

```bash
cd Bert4Rec
bash finetune.sh
```

You need to set the type of initialized item embeddings `--init_type` (adopting randomly initialized embedding if set to None), evaluation setting `--sample_size` (adopting random sampling setting if sample_size is set to be greater than 0, adopting full-ranking setting if sample_size is set to be lower than 0). And for BERT4Rec, you could change its training setting in templates.py

## Pre-training from scratch

If you want to reproduce our results from scratch, we provide detailed implementation procedures here (the same as the procedure in RECFORMER's [repository](https://github.com/AaronHeee/RecFormer)). However, it should be noted that we used different pre-training datasets with RECFORMER, leading to two different pre-trained models. You could also use their pre-trained model to further fine-tune and validate the effectiveness of our method.

First, you need to adjust the original Longformer checkpoint (allenai/longformer-base-4096) to the RECFORMER. You can run the following command:

```bash
cd Recformer
python save_longformer_ckpt.py
```

This code will automatically download `allenai/longformer-base-4096` from Huggingface then adjust and save it to `longformer_ckpt/longformer-base-4096.bin`. 

Then, you can pre-train the RECFORMER by running the following command:

```
bash lightning_run.sh
```

If you use the training strategy `deepspeed_stage_2` (default setting in the script), you need to first convert zero checkpoint to lightning checkpoint by running `zero_to_fp32.py` (automatically generated to checkpoint folder from PyTorch-lightning):

```
python zero_to_fp32.py . pytorch_model.bin
```

Finally, please convert the lightning checkpoint to the Pytorch checkpoint (they have different model parameter names) by running `convert_pretrain_ckpt.py`:

```
python convert_pretrain_ckpt.py
```

You need to set four paths in the file:

- `LIGHTNING_CKPT_PATH`, pre-trained lightning checkpoint path.
- `LONGFORMER_CKPT_PATH`, Longformer checkpoint (from `save_longformer_ckpt.py`) path.
- `OUTPUT_CKPT_PATH`, output path of Recformer checkpoint (for class `RecformerModel` in `recformer/models.py`).
- `OUTPUT_CONFIG_PATH`, output path of Recformer for Sequential Recommendation checkpoint (for class `RecformerForSeqRec` in `recformer/models.py`).

## Acknowledgment

The implementation is based on the open-source repositories [BERT4Rec](https://github.com/SungMinCho/BERT4Rec-PyTorch), [SASRec](https://github.com/pmixer/SASRec.pytorch), and [RECFORMER](https://github.com/AaronHeee/RecFormer).

Please cite our paper as a reference if you use our codes or the processed datasets.

```bibtex
@article{qu2024elephant,
  title={The Elephant in the Room: Rethinking the Usage of Pre-trained Language Model in Sequential Recommendation},
  author={Qu, Zekai and Xie, Ruobing and Xiao, Chaojun and Sun, Xingwu and Kang, Zhanhui},
  journal={arXiv preprint arXiv:2404.08796},
  year={2024}
}
```
