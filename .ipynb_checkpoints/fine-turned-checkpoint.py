import os
os.environ['WANDB_NOTEBOOK_NAME'] = 'sungbohsun'
import torch
import wandb
import argparse
import warnings
warnings.filterwarnings("ignore")
from model import *
from model_bert import *
from dataloader import *
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import ConcatDataset,TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

    
    
class MIX(nn.Module):

    def __init__(self,model_cnn,model_lyrics):
        super(MIX, self).__init__()
        
        parms = {'sample_rate':44100,
         'window_size':1024,
         'hop_size':320,
         'mel_bins':64,
         'fmin':50,
         'fmax':14000,
         'classes_num':4}
        
        model_lyr = eval(model_lyrics+'()')
        if args.model2 == 'BERT':
            self.lyrics = model_lyr.encoder.bert
        
        if args.model2 == 'ALBERT':
            self.lyrics = model_lyr.encoder.albert      
        
        self.audio  = eval(model_cnn+'(**parms)')        
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
    
def train_class(model,epoch):
    model.train()
    t_loss = 0
    all_prediction = []
    all_label = []
    for batch_idx, (audio,lyrics,target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        optimizer.zero_grad()
        audio = audio.to(device)
        lyrics = lyrics.to(device)
        target = target.to(device)
        output = model(audio,lyrics)
        loss = loss_fn(output, target) #the loss functions expects a batchSizex10 input
        loss.backward()
        optimizer.step()
        t_loss += loss.detach().cpu()
        all_prediction.extend(output.argmax(axis=1).cpu().detach().numpy())
        all_label.extend(target.cpu().detach().numpy())

    f1 = f1_score(all_label,all_prediction, average = 'micro')
    wandb.log({"train_loss": t_loss / len(train_loader)})
    wandb.log({"train_f1": f1})
    print('Train Epoch {} : train_loss : {:.5f} train_f1 : {:.5f}'.format(epoch,t_loss/len(train_loader),f1),end=' ')
    
def test_class(model,epoch):
    model.eval()
    t_loss = 0
    all_prediction = []
    all_label = []
    for batch_idx, (audio,lyrics,target) in enumerate(test_loader):
        audio = audio.to(device)
        lyrics = lyrics.to(device)
        target = target.to(device)
        output = model(audio,lyrics)
        loss = loss_fn(output, target) #the loss functions expects a batchSizex10 input
        t_loss += loss.detach().cpu()
        all_prediction.extend(output.argmax(axis=1).cpu().detach().numpy())
        all_label.extend(target.cpu().detach().numpy())
        
    f1 = f1_score(all_label,all_prediction,average = 'micro')
    wandb.log({"val_loss": t_loss / len(test_loader)})
    wandb.log({"val_f1": f1})
    print('val_loss : {0:.5f} val_f1 : {1:.5f}'.format(t_loss / len(test_loader),f1),end=' ')
    return f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",default='l', help="s,m,l")
    parser.add_argument("--bt", type=int,default=4 ,help="batch size")
    parser.add_argument("--data",default='pq',help='dataset')
    parser.add_argument("--path",help='weight path')
    args = parser.parse_args()

    #args.path = './model/Lyr/MIX_Cnn6_ALBERT_l_2_pq/best_net.pt'
    args.model1 = args.path.split('/')[-2].split('_')[1]
    args.model2 = args.path.split('/')[-2].split('_')[2]
    
    if args.model2 == 'BERT':
        pretrain_tk = 'bert-base-uncased'

    elif args.model2 == 'ALBERT':
        pretrain_tk = 'albert-base-v2'
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1
    all_set = Bimix_dataset(list(range(133)),pretrain_tk)
    kfold = KFold(n_splits=5, shuffle=True)
    
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_set)):
        model = MIX(args.model1,args.model2)  
        model.to(device)
        model.load_state_dict(torch.load(args.path))
        optimizer = torch.optim.Adam(model.parameters(),lr=2e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.95, last_epoch=-1)
        wandb.init(tags=[args.model1,args.model2,args.size,str(args.bt)])
        save_path = 'FT_Kfold{}_{}_{}_{}_{}'.format(fold,args.model1,args.model2,args.size,str(args.bt))
        if not os.path.isdir('model/'+save_path):
            os.mkdir('model/'+save_path)
        wandb.run.name = save_path
        wandb.watch(model)
        train_set = Bimix_dataset(train_ids,pretrain_tk)
        train_loader = DataLoader(train_set, batch_size = 6)
        test_set = Bimix_dataset(test_ids,pretrain_tk)
        test_loader = DataLoader(test_set, batch_size = 6)
        print('--- In fold',fold)
        for epoch in range(1, 300):
            scheduler.step()
            train_class(model,epoch)
            f1 = test_class(model, epoch)
            wandb.log({"lr": scheduler.get_last_lr()[0]})
            print('lr: {:.8f}'.format(scheduler.get_last_lr()[0]))
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(),'model/'+save_path+'/best_net.pt')