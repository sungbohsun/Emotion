import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer

def two_labal(tensor,L):
    if L == 'Va':
        tensor[tensor==3] = 0
        tensor[tensor==2] = 1
    elif L == 'Ar':
        tensor[tensor==1] = 0
        tensor[tensor==3] = 1
        tensor[tensor==2] = 1
    #print(tensor)
    return tensor


class PMEdataset(Dataset):
    
    def __init__ (self,folderList):
        df = pd.read_csv('./data/PME_meta.csv',index_col=0)
        
        self.ids = []
        self.labels = []
        
        self.sr = 44100
        self.max_sec = 60
        self.hop_len = 1

        self.dic  = torch.load('./data/dic_PME.pt')
        
        for i in folderList:
            self.ids.append(df.iloc[i,:]['song_id'])
            self.labels.append(df.iloc[i,:]['label'])
            
        
    def __getitem__(self,index):

        sound = self.dic[str(self.ids[index])]
        soundData = torch.mean(sound[0], dim=0).unsqueeze(1)
        
        tempData = torch.zeros([self.sr*self.max_sec, 1]) #tempData accounts for audio clips that are too short
        if soundData.numel() < self.sr*self.max_sec:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:self.sr*self.max_sec]
            
        soundData = tempData

        soundFormatted = torch.zeros([int(self.sr*self.max_sec/self.hop_len),1])
        soundFormatted[:int(self.sr*self.max_sec/self.hop_len)] = soundData[::self.hop_len] #take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        
        return torch.tensor(soundFormatted[0],dtype=torch.float), self.labels[index]-1
            
    
    def __len__(self):
        return len(self.ids)
    
class Q4dataset(Dataset):
    
    def __init__ (self,folderList):
        df = pd.read_csv('./data/Q4_meta.csv',index_col=0)
        
        self.ids = []
        self.labels = []
        
        self.sr = 44100
        self.max_sec = 60
        self.hop_len = 1

        self.dic  = torch.load('./data/dic_Q4.pt')
        
        for i in folderList:
            self.ids.append(df.iloc[i,:]['song_id'])
            self.labels.append(df.iloc[i,:]['label'])
            
        
    def __getitem__(self,index):

        sound = self.dic[str(self.ids[index])]
        soundData = torch.mean(sound[0], dim=0).unsqueeze(1)
        
        tempData = torch.zeros([self.sr*self.max_sec, 1]) #tempData accounts for audio clips that are too short
        if soundData.numel() < self.sr*self.max_sec:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:self.sr*self.max_sec]
            
        soundData = tempData

        soundFormatted = torch.zeros([int(self.sr*self.max_sec/self.hop_len),1])
        soundFormatted[:int(self.sr*self.max_sec/self.hop_len)] = soundData[::self.hop_len] #take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        
        return torch.tensor(soundFormatted[0],dtype=torch.float), self.labels[index]-1
            
    
    def __len__(self):
        return len(self.ids)
    
class DEAMdataset(Dataset):
    
    def __init__ (self,folderList):
        df = pd.read_csv('./data/DEAM_meta.csv',index_col=0)
        
        self.ids = []
        self.labels = []
        
        self.sr = 44100
        self.max_sec = 60
        self.hop_len = 1

        self.dic = torch.load('./data/dic_DEAM.pt')
        
        for i in folderList:
            self.ids.append(df.iloc[i,:]['song_id'])
            self.labels.append(df.iloc[i,:]['label'])
            
        
    def __getitem__(self,index):

        sound = self.dic[str(self.ids[index])]
        soundData = torch.mean(sound[0], dim=0).unsqueeze(1)
        
        tempData = torch.zeros([self.sr*self.max_sec, 1]) #tempData accounts for audio clips that are too short
        if soundData.numel() < self.sr*self.max_sec:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:self.sr*self.max_sec]
            
        soundData = tempData

        soundFormatted = torch.zeros([int(self.sr*self.max_sec/self.hop_len),1])
        soundFormatted[:int(self.sr*self.max_sec/self.hop_len)] = soundData[::self.hop_len] #take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        
        return torch.tensor(soundFormatted[0],dtype=torch.float), self.labels[index]-1
            
    
    def __len__(self):
        return len(self.ids)
    
class PMEmix_dataset(Dataset):
    
    def __init__ (self,folderList,pretrain_tk):
        
        self.df = pd.read_csv('./data/PME_meta.csv',index_col=0)
        self.dic_audio  = torch.load('./data/dic_PME.pt')
        self.dic_lyrics  = torch.load('./data/dic_PME_lyrics.pt')
        
        self.ids = []
        self.labels = []
        
        self.sr = 44100
        self.max_sec = 60
        self.hop_len = 1
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_tk)
        
        for i in folderList:
            ids = sorted(list(self.dic_lyrics.keys()))[i]
            if ids in list(self.dic_audio.keys()):
                self.ids.append(ids)
                  
    def __getitem__(self,index):

        sound = self.dic_audio[str(self.ids[index])]
        soundData = torch.mean(sound[0], dim=0).unsqueeze(1)
        
        tempData = torch.zeros([self.sr*self.max_sec, 1]) #tempData accounts for audio clips that are too short
        if soundData.numel() < self.sr*self.max_sec:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:self.sr*self.max_sec]
            
        soundData = tempData

        soundFormatted = torch.zeros([int(self.sr*self.max_sec/self.hop_len),1])
        soundFormatted[:int(self.sr*self.max_sec/self.hop_len)] = soundData[::self.hop_len] #take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        label = torch.Tensor(self.df[self.df['song_id'] == int(self.ids[index])]['label'].values)[0]
        lyrics = torch.Tensor(self.tokenizer.encode(self.dic_lyrics[self.ids[index]] ,add_special_tokens=True, padding='max_length', max_length=512)[:512])
        
        return torch.tensor(soundFormatted[0],dtype=torch.float), torch.tensor(lyrics,dtype=torch.long),torch.tensor(label,dtype=torch.long)-1
    
        
    def __len__(self):
        return len(self.ids)
    
    
class Q4mix_dataset(Dataset):
    
    def __init__ (self,folderList,pretrain_tk):
        
        self.df = pd.read_csv('./data/Q4_meta.csv',index_col=0)
        self.dic_audio  = torch.load('./data/dic_Q4.pt')
        self.dic_lyrics  = torch.load('./data/dic_Q4_lyrics.pt')
        
        self.ids = []
        self.labels = []
        
        self.sr = 44100
        self.max_sec = 60
        self.hop_len = 1
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_tk)
        
        for i in folderList:
            ids = sorted(list(self.dic_lyrics.keys()))[i]
            if ids in list(self.dic_audio.keys()):
                self.ids.append(ids)
                  
    def __getitem__(self,index):

        sound = self.dic_audio[str(self.ids[index])]
        soundData = torch.mean(sound[0], dim=0).unsqueeze(1)
        
        tempData = torch.zeros([self.sr*self.max_sec, 1]) #tempData accounts for audio clips that are too short
        if soundData.numel() < self.sr*self.max_sec:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:self.sr*self.max_sec]
            
        soundData = tempData

        soundFormatted = torch.zeros([int(self.sr*self.max_sec/self.hop_len),1])
        soundFormatted[:int(self.sr*self.max_sec/self.hop_len)] = soundData[::self.hop_len] #take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        label = torch.Tensor(self.df[self.df['song_id'] == self.ids[index]]['label'].values)[0]
        lyrics = torch.Tensor(self.tokenizer.encode(self.dic_lyrics[self.ids[index]] ,add_special_tokens=True, padding='max_length', max_length=512)[:512])
        
        return torch.tensor(soundFormatted[0],dtype=torch.float), torch.tensor(lyrics,dtype=torch.long),torch.tensor(label,dtype=torch.long)-1
    
        
    def __len__(self):
        return len(self.ids)
    
class Bimix_dataset(Dataset):
    
    def __init__ (self,folderList,pretrain_tk):
        
        self.df = pd.read_csv('./data/Bi_meta.csv',index_col=0)
        self.dic_audio  = torch.load('./data/dic_Bi.pt')
        self.dic_lyrics  = torch.load('./data/dic_Bi_lyrics.pt')
        
        self.ids = []
        self.labels = []
        
        self.sr = 44100
        self.max_sec = 60
        self.hop_len = 1
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_tk)
        
        for i in folderList:
            ids = sorted(list(self.dic_lyrics.keys()))[i]
            
            if self.df[self.df['Code']==ids]['Song Code'].iloc[0] in list(self.dic_audio.keys()):
                self.ids.append(ids)
                  
    def __getitem__(self,index):
        sound = self.dic_audio[str(self.df[self.df['Code']==str(self.ids[index])]['Song Code'].iloc[0])]
        soundData = torch.mean(sound[0], dim=0).unsqueeze(1)
        
        tempData = torch.zeros([self.sr*self.max_sec, 1]) #tempData accounts for audio clips that are too short
        if soundData.numel() < self.sr*self.max_sec:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:self.sr*self.max_sec]
            
        soundData = tempData

        soundFormatted = torch.zeros([int(self.sr*self.max_sec/self.hop_len),1])
        soundFormatted[:int(self.sr*self.max_sec/self.hop_len)] = soundData[::self.hop_len] #take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        label = int(self.df[self.df['Code'] == self.ids[index]]['Quadrant'].iloc[0][1:])
        lyrics = torch.Tensor(self.tokenizer.encode(self.dic_lyrics[self.ids[index]] ,add_special_tokens=True, padding='max_length', max_length=512)[:512])
        
        return torch.tensor(soundFormatted[0],dtype=torch.float), torch.tensor(lyrics,dtype=torch.long),torch.tensor(label,dtype=torch.long)-1
    
        
    def __len__(self):
        return len(self.ids)