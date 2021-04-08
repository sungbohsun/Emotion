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

def train_class(model,epoch):
    model.train()
    t_loss = 0
    all_prediction = []
    all_label = []
    for batch_idx, (data,lyrics,target) in tqdm(enumerate(train_loader),total=len(train_loader)):
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
    
def test_class(model):
    model.eval()
    t_loss = 0
    all_prediction = []
    all_label = []
    for data,lyrics,target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data.to(device))['clipwise_output']
        loss = loss_fn(output, target) #the loss functions expects a batchSizex10 input
        t_loss += loss.detach().cpu()
        all_prediction.extend(output.argmax(axis=1).cpu().detach().numpy())
        all_label.extend(target.cpu().detach().numpy())
        
    f1 = f1_score(all_label,all_prediction,average = 'micro')
    print('val_loss : {0:.5f} val_f1 : {1:.5f}'.format(t_loss / len(test_loader),f1),end=' ')
    return all_label,all_prediction


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bt", type=int,default=4 ,help="batch size")
    parser.add_argument("--path",help='weight path')
    args = parser.parse_args()

    #args.path = './model/Lyr/MIX_Cnn6_ALBERT_l_2_pq/best_net.pt'
    args.model1 = args.path.split('/')[-2].split('_')[1]
    #args.model2 = args.path.split('/')[-2].split('_')[2]
    #Emotion/model/Lyr_Cnn6_l_6_pq/best_net.pt
        
#     if args.model2 == 'BERT':
#         pretrain_tk = 'bert-base-uncased'
    
#     elif args.model2 == 'ALBERT':
#         pretrain_tk = 'albert-base-v2'
        
#     else:
    print('--- use Cnn model without lyrics ---')
    pretrain_tk = 'bert-base-uncased'
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    
    test_set = Bimix_dataset(list(range(133)),pretrain_tk)
    
    kwargs = {'num_workers': 5, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu
    test_loader = DataLoader(test_set, batch_size = args.bt ,shuffle = True, **kwargs)
            
    parms = {'sample_rate':44100,
     'window_size':1024,
     'hop_size':320,
     'mel_bins':64,
     'fmin':50,
     'fmax':14000,
     'classes_num':4}
    
    model = eval(args.model1+'(**parms)') 
    model.to(device)
    
    model.load_state_dict(torch.load(args.path))
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.95, last_epoch=-1)
    loss_fn = nn.CrossEntropyLoss()
    
    y_true, y_pred = test_class(model)
    print('Classification Report:')
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred,list(range(4)))
    fig = plt.figure(figsize=(10,10),dpi=80)
    ax = fig.add_subplot(2,1,1)
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    
#     if not os.path.isdir('png/'+args.path):
#         os.mkdir('png/'+args.path)
    fig.savefig('png/CNN_confusion_matrix.png')

