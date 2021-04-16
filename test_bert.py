import os
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
from model_bert import *
from dataloader import *
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import ConcatDataset,TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

def test_class(model,epoch):
    model.eval()
    all_prediction = []
    all_label = []
    for audio,lyrics,target in test_loader:
        target = two_labal(target,args.mode)
        #audio = audio[:,0,:].to(device)
        lyrics = lyrics.to(device) 
        target = target.to(device)
        loss,output = model(lyrics,target)
        all_prediction.extend(output.argmax(axis=1).cpu().detach().numpy())
        all_label.extend(target.cpu().detach().numpy())

    return all_label,all_prediction


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bt", type=int,default=4 ,help="batch size")
    parser.add_argument("--path",help='weight path')
    parser.add_argument("--mode",  help='Va,Ar',default='4Q')
    args = parser.parse_args()
    
    args.model1 = args.path.split('/')[-2].split('_')[2]
    print('--- use Cnn model without lyrics ---')
    pretrain_tk = 'bert-base-uncased'
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    all_set = PMEmix_dataset(list(range(629)),pretrain_tk)
    skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=8848)
    skf.get_n_splits(all_set)
    for fold, (train_index, test_index) in enumerate(skf.split(all_set,[y for _,_,y in all_set])):
        
        print('now is in fold',fold)
        test_set = PMEmix_dataset(test_index,pretrain_tk)
        kwargs = {'num_workers': 5, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu
        test_loader = DataLoader(test_set, batch_size = args.bt ,shuffle = True, **kwargs)

        if args.mode == '4Q':
            model = eval(args.model+'()')   
        else:
            model = eval(args.model+'_TL()')  
            
        model.to(device)

        model.load_state_dict(torch.load(args.path))

        y_true, y_pred = test_class(model)
        print('Classification Report:')
        print(classification_report(y_true, y_pred, digits=4))

        cm = confusion_matrix(y_true, y_pred,list(range(4 if args.mode == '4Q' else 2)))
        fig = plt.figure(figsize=(10,10),dpi=80)
        ax = fig.add_subplot(2,1,1)
        sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

        ax.set_title('Confusion Matrix')

        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')

        fig.savefig('png/Lyrics_{}_{}_fold{}.png'.format(args.mode,args.model1,fold))

