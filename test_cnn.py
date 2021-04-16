import os
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
from model import *
from dataloader import *
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import ConcatDataset,TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
    
def test_class(model):
    model.eval()
    all_prediction = []
    all_label = []
    for data,lyrics,target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data.to(device))['clipwise_output']
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

        parms = {'sample_rate':44100,
         'window_size':1024,
         'hop_size':320,
         'mel_bins':64,
         'fmin':50,
         'fmax':14000,
         'classes_num':4 if args.mode == '4Q' else 2}

        model = eval(args.model1+'(**parms)') 
        model.to(device)
        load_path = './model/{}{}/best_net.pt'.format(args.path.split('/')[-2][:-1],fold)
        print('load',load_path)
        model.load_state_dict(torch.load(load_path))

        y_true, y_pred = test_class(model)
        print('Classification Report:')
        report = classification_report(y_true, y_pred, digits=4)
        print(report)

        cm = confusion_matrix(y_true, y_pred,list(range(4 if args.mode == '4Q' else 2)))
        fig = plt.figure(figsize=(10,10),dpi=80)
        ax = fig.add_subplot(2,1,1)
        sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

        ax.set_title('Confusion Matrix')

        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        fig.savefig('png/Audio_{}_{}_fold{}.png'.format(args.mode,args.model1,fold))
        
        f = open('res/{}{}.txt'.format(args.path.split('/')[-2][:-1],fold),'w+')
        f.write(report)
        f.close()

