CUDA_VISIBLE_DEVICES=1, python train_cnn.py --model Cnn6 --mode 4Q --fold 0 --CV ALL
CUDA_VISIBLE_DEVICES=1, python train_cnn.py --model Cnn6 --mode 4Q --fold 1 --CV ALL
CUDA_VISIBLE_DEVICES=1, python train_cnn.py --model Cnn6 --mode 4Q --fold 2 --CV ALL
CUDA_VISIBLE_DEVICES=1, python train_cnn.py --model Cnn6 --mode 4Q --fold 3 --CV ALL
CUDA_VISIBLE_DEVICES=1, python train_cnn.py --model Cnn6 --mode 4Q --fold 4 --CV ALL

# CUDA_VISIBLE_DEVICES=0, python train_cnn.py --model Cnn6 --mode Ar
# CUDA_VISIBLE_DEVICES=0, python train_cnn.py --model Cnn6 --mode Va