for i in  $(seq 0 4);
do
    #echo $i;
    CUDA_VISIBLE_DEVICES=0, python train_mix.py --mode 4Q \
    --path1 ./model/Audio_4Q_Cnn6_fold-${i}/best_net.pt \
    --path2 ./model/Lyrics_4Q_BERT_fold-${i}/best_net.pt
done