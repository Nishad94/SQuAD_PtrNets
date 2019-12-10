import numpy as np
import pandas as pd
import re
import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from nltk.metrics.distance import edit_distance
import string
import matplotlib.pyplot as plt
import pickle

with open('dict_pointer.pkl', 'rb') as handle:
    word2idx = pickle.load(handle)

# function to return vector of int for str
def vec_int(str_ip, count_missing):
    lst_ret = []
    is_invalid = 0
    lst_str_ip = re.findall(r"[\w']+|[.,!?;]", str_ip)
    for word in lst_str_ip:
        if (word not in word2idx):
#            print(word)
            idx = 0
            count_missing += 1
            is_invalid = 1
        else:
            idx = word2idx[word]
        lst_ret.append(idx)
        
    return lst_ret, count_missing, is_invalid

count_missing = 0
valStartIndex = 20000
valExamples = 2000

class dataset(Dataset):
    def __init__(self, data_dir):
        super(dataset, self).__init__()
        self.df = data_dir
                
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i): # return single data item
        dict_ret = {}
        

        answerWindow = [int(self.df['Answer'][i][0][0]), int(self.df['Answer'][i][0][1]) - 1]
        lst_quest, _, _ = vec_int(self.df['Question'][i], count_missing)
        lst_context, _, _ = vec_int(self.df['Context'][i], count_missing)
        
        dict_ret['Question_Txt'] = self.df['Question'][i]
        dict_ret['Question_Tensor'] = torch.LongTensor(lst_quest)
        dict_ret['Context_Txt'] = self.df['Context'][i]
        dict_ret['Context_Tensor'] = torch.LongTensor(lst_context)
        dict_ret['Answer'] = torch.LongTensor(answerWindow)
#        print (type(dict_ret))
        return dict_ret

class val_dataset(Dataset):
    def __init__(self, data_dir):
        super(val_dataset, self).__init__()
        self.df = data_dir
                
    def __len__(self):
        return len(self.df)

    def __getitem__(self, old_i): # return single data item
        dict_ret = {}
        i = old_i + valStartIndex

        answerWindow = [int(self.df['Answer'][i][0][0]), int(self.df['Answer'][i][0][1]) - 1]
        lst_quest, _, _ = vec_int(self.df['Question'][i], count_missing)
        lst_context, _, _ = vec_int(self.df['Context'][i], count_missing)
        
        dict_ret['Question_Txt'] = self.df['Question'][i]
        dict_ret['Question_Tensor'] = torch.LongTensor(lst_quest)
        dict_ret['Context_Txt'] = self.df['Context'][i]
        dict_ret['Context_Tensor'] = torch.LongTensor(lst_context)
        dict_ret['Answer'] = torch.LongTensor(answerWindow)
#        print (type(dict_ret))
        return dict_ret



df_format_full = pd.read_pickle("processed_data.pkl")
df_format_final = df_format_full.iloc[0:valStartIndex, :]
val_df = df_format_full.iloc[valStartIndex:valStartIndex + valExamples, :]

train_data = dataset(df_format_final)
val_data = val_dataset(val_df)
# test_data = dataset(df_format)


# create train and test dataloader objects
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 1, shuffle = False) 
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 1, shuffle = False) 
#test_loader = torch.utils.data.DataLoader(test_data, batch_size = bs, collate_fn = collate, shuffle = False) 

# uncomment below for testing
# for index, (df) in enumerate(train_loader):
#     questionText = df['Question_Txt']
#     questionTensor = df["Question_Tensor"]
#     contextText = df["Context_Txt"]
#     contextTensor = df['Context_Tensor']
#     answer = df['Answer']
#     print (df)
#     break
