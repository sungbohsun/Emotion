import os
os.environ['WANDB_NOTEBOOK_NAME'] = 'sungbohsun'
import torch
import wandb
import argparse
import warnings
warnings.filterwarnings("ignore")
from model import *
from dataloader import *
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import ConcatDataset,TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def train_class(model,epoch):
    model.train()
    t_loss = 0
    all_prediction = []
    all_label = []
    for batch_idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)['clipwise_output']
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
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)['clipwise_output']
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
    parser.add_argument("--size", help="s,m,l")
    parser.add_argument("--bt", type=int ,help="batch size")
    parser.add_argument("--model",help='Cnn6')
    parser.add_argument("--data",help='dataset')
    args = parser.parse_args()

    if  args.size == 's': 

        data_size_PME  = 767//50
        data_size_Q4   = 900//50
        data_size_DEAM = 1802//50
    
    elif args.size == 'm': 
        
        data_size_PME  = 767//10
        data_size_Q4   = 900//10
        data_size_DEAM = 1802//10
    
    elif args.size == 'l': 
        
        data_size_PME  = 767
        data_size_Q4   = 900
        data_size_DEAM = 1802
        
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    train_sets = []
    test_sets  = []
    
    if  args.data.find('p')>=0:
        print('-------- load PME')
        train_index,test_index = train_test_split(list(range(data_size_PME)),test_size=0.2,random_state=7777)
        PME_train_set = PMEdataset(train_index)
        PME_test_set = PMEdataset(test_index)
        train_sets.append(PME_train_set)
        test_sets.append(PME_test_set)
    
    if  args.data.find('q')>=0:
        print('-------- load Q4')
        train_index,test_index = train_test_split(list(range(data_size_Q4)),test_size=0.2,random_state=7777)
        Q4_train_set = Q4dataset(train_index)
        Q4_test_set = Q4dataset(test_index)
        train_sets.append(Q4_train_set)
        test_sets.append(Q4_test_set)
    
    if  args.data.find('d')>=0:
        print('-------- load DEAM')
        train_index,test_index = train_test_split(list(range(data_size_DEAM)),test_size=0.2,random_state=7777)
        DEAM_train_set = DEAMdataset(train_index)
        DEAM_test_set = DEAMdataset(test_index)
        train_sets.append(DEAM_train_set)   
        test_sets.append(DEAM_test_set)
    
    train_set = ConcatDataset(train_sets)
    test_set = ConcatDataset(test_sets)
    
    kwargs = {'num_workers': 5, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu
    train_loader = DataLoader(train_set, batch_size = args.bt,shuffle = True, **kwargs)
    test_loader = DataLoader(test_set, batch_size = args.bt ,shuffle = True, **kwargs)
    
    parms = {'sample_rate':44100,
             'window_size':1024,
             'hop_size':320,
             'mel_bins':64,
             'fmin':50,
             'fmax':14000,
             'classes_num':4}
    
    model = eval(args.model+'(**parms)')   
    model.to(device)
    wandb.init(tags=[args.model,args.size,str(args.bt)])
    save_path = '{}_{}_{}_{}'.format(args.model,args.size,str(args.bt),args.data)
    wandb.run.name = save_path
    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.95, last_epoch=-1)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1
    
    if not os.path.isdir('model/'+save_path):
        os.mkdir('model/'+save_path)
        
    for epoch in range(1, 300):
        scheduler.step()
        train_class(model,epoch)
        f1 = test_class(model, epoch)
        wandb.log({"lr": scheduler.get_last_lr()[0]})
        print('lr: {:.8f}'.format(scheduler.get_last_lr()[0]))
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(),'model/'+save_path+'/best_net.pt')