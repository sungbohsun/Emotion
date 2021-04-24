for i in  $(seq 0 4);
do
    #echo $i;
    CUDA_VISIBLE_DEVICES=3, python test_cnn.py --path ./model/Audio_4Q_Cnn6_fold-${i}/best_net.pt
done

for i in  $(seq 0 4);
do
    #echo $i;
    CUDA_VISIBLE_DEVICES=3, python test_cnn.py --path ./model/Audio_4Q_Cnn10_fold-${i}/best_net.pt
done