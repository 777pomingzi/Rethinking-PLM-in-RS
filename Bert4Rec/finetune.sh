CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.run \
                            --nproc_per_node=1 --master_port='29502' main.py
