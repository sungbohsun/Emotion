import os
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
from model import *
from model_bert import *
from dataloader import *
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import ConcatDataset,TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
    
class MIX(nn.Module):

    def __init__(self,model_cnn,model_lyrics):
        super(MIX, self).__init__()
        
        parms = {'sample_rate':44100,
         'window_size':1024,
         'hop_size':320,
         'mel_bins':64,
         'fmin':50,
         'fmax':14000,
         'classes_num':4 if args.mode == '4Q' else 2}
        
        if args.mode == '4Q':
            model_lyr = eval(model_lyrics+'()')
        else:
            model_lyr = eval(model_lyrics+'_TL()')
        
        if args.model2 == 'BERT':
            model_lyr.load_state_dict(torch.load(args.path2))
            self.lyrics = model_lyr.encoder.bert
        
        if args.model2 == 'ALBERT':
            model_lyr.load_state_dict(torch.load(args.path2))
            self.lyrics = model_lyr.encoder.albert
        
        self.audio  = eval(model_cnn+'(**parms)')
        
        if args.model1 == 'Cnn6':
            self.audio.load_state_dict(torch.load(args.path1))
        
        if args.model1 == 'Cnn10':
            self.audio.load_state_dict(torch.load(args.path1))
        
        self.audio.fc_audioset = layer_pass()
        self.fc1 = nn.Linear(768+512,768+512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(768+512,4)
    def forward(self,x1,x2):
        
#         for param in self.audio.parameters():
#             param.requires_grad = False
        
#         for param in self.lyrics.parameters():
#             param.requires_grad = False
        
        x2 = torch.tensor(x2, dtype=torch.long)
        out1 = self.audio(x1)['clipwise_output']
        out2 = self.lyrics(x2)['pooler_output']
        out = torch.cat((out1,out2),dim=1)
        out = self.drop(out)
        out = self.fc1(out)
        result = self.fc2(out)
        
        return result
    
def test_class(model):
    model.eval()
    all_prediction = []
    all_label = []
    for batch_idx, (audio,lyrics,target) in enumerate(test_loader):
        audio = audio.to(device)
        lyrics = lyrics.to(device)
        target = target.to(device)
        output = model(audio,lyrics)
        all_prediction.extend(output.argmax(axis=1).cpu().detach().numpy())
        all_label.extend(target.cpu().detach().numpy())
    
    return all_label,all_prediction


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bt", type=int,default=4 ,help="batch size")
    parser.add_argument("--path",help='weight path')
    parser.add_argument("--mode",  help='Va,Ar',default='4Q')
    args = parser.parse_args()

    #args.path = './model/Lyr/MIX_Cnn6_ALBERT_l_2_pq/best_net.pt'
    args.model1 = args.path.split('/')[-2].split('_')[2]
    args.model2 = args.path.split('/')[-2].split('_')[3]
        
    if args.model2 == 'BERT':
        pretrain_tk = 'bert-base-uncased'
    
    elif args.model2 == 'ALBERT':
        pretrain_tk = 'albert-base-v2'
        
    else:
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

        model = MIX(args.model1,args.model2) 
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

        fig.savefig('png/MIX_{}_{}_{}_fold{}.png'.format(args.mode,args.model1,args.model1,fold))

