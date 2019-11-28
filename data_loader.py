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
from tqdm import tqdm

# dev_set_json = '/Users/mayanksharma/Desktop/SQuAD-v1.1.json'
dev_set_csv = 'SQuAD-v1.1.csv'

data_csv = pd.read_csv(dev_set_csv, encoding = "ISO-8859-1")

# build dict
word2idx = {}
idx = 0
# for i in range(0, 1):
for i in range(0, len(data_csv)):
    # add context vocab to dict
#     print(i)
    context = data_csv['Context'][i]
    context = context.replace("\'s", '')
    context = context.replace("\'", '')
#     lst_words_context = re.findall(r'\w+', context)
    lst_words_context = re.findall(r"[\w']+|[.,!?;]", context)
#     lst_words_context = re.split('[^a-zA-Z0-9]', context)
    for word in lst_words_context:
        if word not in word2idx:
#             if (word[0] == '\'' and word[-1] == '\''):
#                 word = word[1:-1]
            word2idx[word] = idx
            idx += 1
    
    # add question vocab to dict 
    # loop across number of questions for a give context
    for j in range(len(data_csv['QuestionAnswerSets'][i].split("|\"Question\"")[1:])):
        question = data_csv['QuestionAnswerSets'][i].split("|\"Question\"")[1:][j].split("->")[1][2:-14]
        lst_words_question = re.findall(r"[\w']+|[.,!?;]", question)
        for word in lst_words_question:
            if word not in word2idx:
#                 if (word[0] == '\''):
#                     word = word[1:-1]
                word2idx[word] = idx
                idx += 1
        
# function to return vector of int for str
def vec_int(str_ip, count_missing):
    lst_ret = []
    is_invalid = 0
    lst_str_ip = re.findall(r"[\w']+|[.,!?;]", str_ip)
    for word in lst_str_ip:
        if (word not in word2idx):
            print(word)
            idx = 0
            count_missing += 1
            is_invalid = 1
        else:
            idx = word2idx[word]
        lst_ret.append(idx)
        
    return lst_ret, count_missing, is_invalid

def get_starting_index(lst1, lst2):
    for i in range(0, len(lst1)):
        if (lst1[i:i+len(lst2)] == lst2):
            return (i, i+len(lst2)-1)

# create dataframe for getitem
count_missing = 0
df_format = pd.DataFrame(columns = ['Question', 'Context', 'Answer', 'is_invalid'])
df_format_final = pd.DataFrame(columns = ['Question', 'Context', 'Answer'])
# for i in range(0, 50):
print(len(data_csv))
# for i in tqdm(range(len(data_csv))):
for i in tqdm(range(4)):
    context = data_csv['Context'][i]
#     context = context.replace('\'', '')
    context = context.replace("\'s", '')
    context = context.replace("\'", '')
    context_vec, count_missing, is_invalid = vec_int(context, count_missing)
    if (is_invalid == 1):
        continue
    qa_list = data_csv['QuestionAnswerSets'][i].split("|\"Question\"")[1:]
    for j in range(0, len(qa_list)):
        question = qa_list[j].split("->")[1][2:-14]
        question = question.replace('\'', '')
        answer = qa_list[j].split("->")[2][3:-22]
        answer = answer.replace("\'s", '')
        answer = answer.replace("\'", '')
#         answer = answer.replace('\'', '')
#         df_format['Question'].append(question)
#         df_format['Context'].append(context)
        
        # get start and end indices for answer
#         s_idx = str(int(data_csv['QuestionAnswerSets'][i].split("|\"Question\"")[1:][j].split("->")[3][2:-16]) - 1)
#         e_idx = str(int(s_idx) + len(data_csv['QuestionAnswerSets'][i].split("|\"Question\"")[1:][j].split("->")[2][3:-22]))
        
        
        answer_vec, count_missing, is_invalid = vec_int(answer, count_missing)
        
        ans_idx = get_starting_index(context_vec, answer_vec)
        
#         df_format['Answer'].append((s_idx, e_idx))
        df_format = df_format.append({'Question' : question, 'Context' : context, 'Answer': ans_idx, 'is_invalid': is_invalid} , ignore_index = True)
        if (is_invalid == 0 and ans_idx is not None):
            df_format_final = df_format_final.append({'Question' : question, 'Context' : context, 'Answer': ans_idx} , ignore_index = True)
            

class dataset(Dataset):
    def __init__(self, data_dir):
        super(dataset, self).__init__()
        self.df = data_dir
                
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i): # return single data item
        # if i == 27:
        #     import pdb
        #     pdb.set_trace()

        answerWindow = [int(self.df['Answer'][i][0]), int(self.df['Answer'][i][1])]
#   print (answerWindow)
        questionList, _, _ = vec_int(self.df['Question'][i], count_missing)
        contextList,_,_ = vec_int(self.df['Context'][i], count_missing)
        dict_ret = {}
        dict_ret['Question_Txt'] = self.df['Question'][i]
        dict_ret['Question_Tensor'] = torch.Tensor(questionList)
        dict_ret['Context_Txt'] = self.df['Context'][i]
        dict_ret['Context_Tensor'] = torch.Tensor(contextList)
        dict_ret['Answer'] = torch.LongTensor(answerWindow)
        return dict_ret
        # return torch.Tensor(questionList), torch.Tensor(contextList), torch.LongTensor(answerWindow)
    

train_data = dataset(df_format_final)
test_data = dataset(df_format_final)


# create train and test dataloader objects
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 1, shuffle = True) 
#test_loader = torch.utils.data.DataLoader(test_data, batch_size = bs, collate_fn = collate, shuffle = False) 

# for index, (df) in enumerate(train_loader):
#     question = df[0]
#     context = df[1]
#     answer = df[2]
#     break