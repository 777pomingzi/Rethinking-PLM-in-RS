
python finetune.py \
    --data_path finetune_data/Arts \
    --num_train_epochs 128 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --device 6 \
    --fp16 \
    --finetune_negative_sample_size -1 \
    --ckpt best_model.bin \
    --ckptf1 best_model_1.bin \
    --pretrain_ckpt pretrain_ckpt/seqrec_pretrain_ckpt.bin \
    --longformer_ckpt longformer_ckpt/longformer-base-4096.bin \
    --sample_size 100 \
    --test_only False 
    