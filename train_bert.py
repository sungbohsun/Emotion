import os
os.environ['WANDB_NOTEBOOK_NAME'] = 'sungbohsun'
import torch
import wandb
import argparse
import warnings
warnings.filterwarnings("ignore")
from model_bert import *
from dataloader import *
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import ConcatDataset,TensorDataset
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import StratifiedKFold

def eval_score(all_label,all_prediction,EPT):
            
    f1_micro = f1_score(all_label,all_prediction, average = 'micro')
    f1_marco = f1_score(all_label,all_prediction, average = 'macro')
    f1_weighted = f1_score(all_label,all_prediction, average = 'weighted')
    acc = accuracy_score(all_label,all_prediction)
    if EPT == 'train':
        wandb.log({"train_f1_micro":    f1_micro})
        wandb.log({"train_f1_marco":    f1_marco})
        wandb.log({"train_f1_weighted": f1_weighted})
        wandb.log({"train_acc": acc})

    if EPT == 'test':
        wandb.log({"val_f1_micro":    f1_micro})
        wandb.log({"val_f1_marco":    f1_marco})
        wandb.log({"val_f1_weighted": f1_weighted})
        wandb.log({"val_acc": acc})
    return f1_micro

def train_class(model,epoch):
    model.train()
    t_loss = 0
    all_prediction = []
    all_label = []
    for batch_idx, (audio,lyrics,target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        target = two_labal(target,args.mode)
        optimizer.zero_grad()
        lyrics = lyrics.to(device)
        target = target.to(device)
        loss,output = model(lyrics,target)
        loss.backward()
        optimizer.step()
        t_loss += loss.detach().cpu()
        all_prediction.extend(output.argmax(axis=1).cpu().detach().numpy())
        all_label.extend(target.cpu().detach().numpy())
        
    wandb.log({"train_loss": t_loss / len(train_loader)})  
    f1 = eval_score(all_label,all_prediction,'train')
    print('Train Epoch {} : train_loss : {:.5f} train_f1 : {:.5f}'.format(epoch,t_loss/len(train_loader),f1),end=' ')
    
def test_class(model,epoch):
    model.eval()
    t_loss = 0
    all_prediction = []
    all_label = []
    for audio,lyrics,target in test_loader:
        target = two_labal(target,args.mode)
        #audio = audio[:,0,:].to(device)
        lyrics = lyrics.to(device) 
        target = target.to(device)
        loss,output = model(lyrics,target)
        t_loss += loss.detach().cpu()
        all_prediction.extend(output.argmax(axis=1).cpu().detach().numpy())
        all_label.extend(target.cpu().detach().numpy())
        
    wandb.log({"val_loss": t_loss / len(train_loader)})       
    f1 = eval_score(all_label,all_prediction,'test')
    print('val_loss : {0:.5f} val_f1 : {1:.5f}'.format(t_loss / len(test_loader),f1),end=' ')
    return f1

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help='BERT')
    parser.add_argument("--bt",    help="batch size",type=int,default=6)
    parser.add_argument("--data",  help='dataset', default='pqb')
    parser.add_argument("--mode",  help='Va,Ar',   default='4Q')
    parser.add_argument("--fold",  help='0-4',type=int)
    parser.add_argument("--CV",    help='PME,ALL',default='PME')
    args = parser.parse_args()

    data_size_Bi  = 133 
    data_size_Q4  = 479
    data_size_PME = 629
    #data_size_DEAM = 1802
        
    if args.model == 'BERT':
        pretrain_tk = 'bert-base-uncased'
    
    elif args.model == 'ALBERT':
        pretrain_tk = 'albert-base-v2'
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    train_sets = []
    test_sets  = []
    
    if args.CV == 'PME' :
        print('CV in PMEmo')
        all_set = PMEmix_dataset(list(range(data_size_PME)),pretrain_tk)
        skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=8848)
        skf.get_n_splits(all_set)

        for fold, (train_index, test_index) in enumerate(skf.split(all_set,[y for _,_,y in all_set])):

            if fold != args.fold: continue

            print('now is in fold',fold)

            if  args.data.find('p')>=0:
                print('-------- load Lyr_PME')
                PME_train_set = PMEmix_dataset(train_index,pretrain_tk)
                PME_test_set = PMEmix_dataset(test_index,pretrain_tk)
                train_sets.append(PME_train_set)
                test_sets.append(PME_test_set)

            if  args.data.find('q')>=0:
                print('-------- load Lyr_Q4')
                Q4_train_set = Q4mix_dataset(list(range(data_size_Q4)),pretrain_tk)
                train_sets.append(Q4_train_set)

            if  args.data.find('b')>=0:
                print('-------- load Bi')
                Bi_train_set = Bimix_dataset(list(range(data_size_Bi)),pretrain_tk)
                train_sets.append(Bi_train_set)

    elif args.CV == 'ALL' :
        print('CV in ALL data')
        all_set = PMEmix_dataset(list(range(data_size_PME)),pretrain_tk)
        skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=8848)
        skf.get_n_splits(all_set)

        for fold, (train_index, test_index) in enumerate(skf.split(all_set,[y for _,_,y in all_set])):

            if fold != args.fold: continue

            print('now is in fold',fold)

            if  args.data.find('p')>=0:
                print('-------- load Lyr_PME')
                PME_train_set = PMEmix_dataset(train_index,pretrain_tk)
                PME_test_set = PMEmix_dataset(test_index,pretrain_tk)
                train_sets.append(PME_train_set)
                test_sets.append(PME_test_set)

        all_set = Q4mix_dataset(list(range(data_size_Q4)),pretrain_tk)
        skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=8848)
        skf.get_n_splits(all_set)

        for fold, (train_index, test_index) in enumerate(skf.split(all_set,[y for _,_,y in all_set])):

            if fold != args.fold: continue      
            if  args.data.find('q')>=0:
                print('-------- load Lyr_Q4')
                Q4_train_set = Q4mix_dataset(train_index,pretrain_tk)
                Q4_test_set = Q4mix_dataset(test_index,pretrain_tk)               
                train_sets.append(Q4_train_set)
                test_sets.append(Q4_test_set)

        all_set = Bimix_dataset(list(range(data_size_Bi)),pretrain_tk)
        skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=8848)
        skf.get_n_splits(all_set)

        for fold, (train_index, test_index) in enumerate(skf.split(all_set,[y for _,_,y in all_set])):

            if fold != args.fold: continue
            if  args.data.find('b')>=0:
                print('-------- load Bi')
                Bi_train_set = Bimix_dataset(train_index,pretrain_tk)
                Bi_test_set = Bimix_dataset(test_index,pretrain_tk)               
                train_sets.append(Bi_train_set)
                test_sets.append(Bi_test_set)

    train_set = ConcatDataset(train_sets)
    test_set = ConcatDataset(test_sets)

    kwargs = {'num_workers': 5, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu
    train_loader = DataLoader(train_set, batch_size = args.bt,shuffle = True, **kwargs)
    test_loader = DataLoader(test_set, batch_size = args.bt ,shuffle = True, **kwargs)


    if args.mode == '4Q':
        model = eval(args.model+'()')   
    else:
        model = eval(args.model+'_TL()')  

    model.to(device)
    wandb.init(tags=[args.CV,args.mode,args.model,'bt-'+str(args.bt),'fold-'+str(args.fold)])
    save_path = '{}-Lyrics_{}_{}_fold-{}'.format(args.CV,args.mode,args.model,args.fold)
    wandb.run.name = save_path
    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(),lr=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.95, last_epoch=-1)

    best_f1 = -1

    if not os.path.isdir('model/'+save_path):
        os.mkdir('model/'+save_path)

    for epoch in range(1, 30):
        scheduler.step()
        train_class(model,epoch)
        f1 = test_class(model, epoch)
        wandb.log({"lr": scheduler.get_last_lr()[0]})
        print('lr: {:.8f}'.format(scheduler.get_last_lr()[0]))
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(),'model/'+save_path+'/best_net.pt')