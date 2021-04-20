CUDA_VISIBLE_DEVICES=1, python train_bert.py --model ALBERT --mode 4Q --fold 0
CUDA_VISIBLE_DEVICES=1, python train_bert.py --model ALBERT --mode 4Q --fold 1
CUDA_VISIBLE_DEVICES=1, python train_bert.py --model ALBERT --mode 4Q --fold 2
CUDA_VISIBLE_DEVICES=1, python train_bert.py --model ALBERT --mode 4Q --fold 3
CUDA_VISIBLE_DEVICES=1, python train_bert.py --model ALBERT --mode 4Q --fold 4

# CUDA_VISIBLE_DEVICES=0, python train_bert.py --model ALBERT --mode Ar
# CUDA_VISIBLE_DEVICES=0, python train_bert.py --model ALBERT --mode Va