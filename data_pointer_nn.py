
# coding: utf-8

import numpy as np
import pandas as pd
import json
import re
import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from nltk.metrics.distance import edit_distance
import string
import matplotlib.pyplot as plt

# data in csv
dev_set_csv = 'SQuAD-v1.1.csv'

# read csv into pandas dataframe
data_csv = pd.read_csv(dev_set_csv, encoding = "ISO-8859-1")

# build dict
word2idx = {}
idx = 0
# for i in range(0, 1):
for i in range(0, len(data_csv)):
    # add context vocab to dict
    # print(i)
    context = data_csv['Context'][i]
    lst_words_context = re.findall(r"[\w']+|[.,!?;]", context)
    for word in lst_words_context:
        if word not in word2idx:
            word2idx[word] = idx
            idx += 1
    
    # add question vocab to dict 
    # loop across number of questions for a give context
    for j in range(len(data_csv['QuestionAnswerSets'][i].split("|\"Question\"")[1:])):
        question = data_csv['QuestionAnswerSets'][i].split("|\"Question\"")[1:][j].split(",")[0][5:-2]
        lst_words_question = re.findall(r"[\w']+|[.,!?;]", question)
        for word in lst_words_question:
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
                

# function to return vector of int for str
def vec_int(str_ip):
    lst_ret = []
    lst_str_ip = re.findall(r"[\w']+|[.,!?;]", str_ip)
    for word in lst_str_ip:
        idx = word2idx[word]
        lst_ret.append(idx)
        
    return lst_ret



# create dataframe for getitem
df_format = pd.DataFrame(columns = ['Question', 'Context', 'Answer'])
for i in range(0, 1000):
# for i in range(0, len(data_csv)):
    context = data_csv['Context'][i]
    for j in range(0, len(data_csv['QuestionAnswerSets'][i].split("|\"Question\"")[1:])):
        question = data_csv['QuestionAnswerSets'][i].split("|\"Question\"")[1:][j].split("->")[1][2:-14]
        
        # get start and end indices for answer
        s_idx = str(int(data_csv['QuestionAnswerSets'][i].split("|\"Question\"")[1:][j].split("->")[3][2:-16]) - 1)
        e_idx = str(int(s_idx) + len(data_csv['QuestionAnswerSets'][i].split("|\"Question\"")[1:][j].split("->")[2][3:-22]))
        
        df_format = df_format.append({'Question' : question , 'Context' : context, 'Answer': (s_idx, e_idx)} , ignore_index = True)
        
class dataset(Dataset):
    def __init__(self, data_dir):
        super(dataset, self).__init__()
        self.df = data_dir
                
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i): # return single data item

        answerWindow = [int(df_format['Answer'][i][0]), int(df_format['Answer'][i][1])]
        return torch.FloatTensor(vec_int(df_format['Question'][i])), torch.FloatTensor(vec_int(df_format['Context'][i])), torch.LongTensor(answerWindow)
    
train_data = dataset(df_format)
# test_data = dataset(df_format)


# create train and test dataloader objects
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 1, shuffle = True) 
#test_loader = torch.utils.data.DataLoader(test_data, batch_size = bs, collate_fn = collate, shuffle = False) 

# for index, (df) in enumerate(train_loader):
#     question = df[0]
#     context = df[1]
#     answer = df[2]
#     break

