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
from EncoderDecoderBaseline import BasicS2S
from data_loader import train_loader, word2idx
from eval import compute_f1
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import argparse
from tqdm import tqdm

from PointerNet import PointerNet
from EncoderDecoderBaseline import BasicS2S
from Data_Generator import TSPDataset

from data_loader import train_loader, word2idx, val_loader
from eval import compute_f1


with open('dict_pointer.pkl', 'rb') as handle:
    word2idx = pickle.load(handle)

revDict = {}

for key, val in word2idx.items():
    revDict[val] = key

def getText(para):
    paragraph = ""
    for p in para:
        paragraph += revDict[p] + " "
    return paragraph

# function to return vector of int for str
def vec_int(str_ip, count_missing):
    lst_ret = []
    is_invalid = 0
    lst_str_ip = re.findall(r"[\w']+|[.,!?;]", str_ip)
    for word in lst_str_ip:
        if (word not in word2idx):
            idx = 0
            count_missing += 1
            is_invalid = 1
        else:
            idx = word2idx[word]
        lst_ret.append(idx)
        
    return lst_ret, count_missing, is_invalid

count_missing = 0

model = BasicS2S(len(word2idx),768,256).cuda()
model.load_state_dict(torch.load("1575985691.8680089_19.pt"))
model.eval()

USE_CUDA = True
valStartIndex = 18500
valExamples = 2000

resultFile = open('results.txt', 'w')
def val_loop_s2s(model,loader):
    iterator = tqdm(loader, unit='Batch')
    total_f1 = 0.0
    goodExamples = 0
    badExamples = 0
    out = ""
    for i_batch, sample_batched in enumerate(iterator):
        #iterator.set_description('Test Batch %i/%i' % (1, params.nof_epoch))
        test_batch_para = Variable(sample_batched["Context_Tensor"]).unsqueeze(2)
        test_batch_quest = Variable(sample_batched["Question_Tensor"]).unsqueeze(2)
        target_batch = Variable(sample_batched["Answer"]).unsqueeze(2)
        test_batch_quest_text = sample_batched["Question_Txt"]
        test_batch_para_text = sample_batched["Context_Txt"]
        #if USE_CUDA:
        test_batch_para = test_batch_para.cuda()
        test_batch_quest = test_batch_quest.cuda()
        # para_len * 3
        o = model(test_batch_para,test_batch_quest, test_batch_para_text, test_batch_quest_text)
        start_probs = o[:,1]
        end_probs = o[:,2]
        start_pos = torch.argmax(start_probs)
        end_pos = torch.argmax(end_probs)
        
        para = test_batch_para.tolist()
        # Flatten 
        para = [l for item in para[0] for l in item]
        score = compute_f1(para[start_pos:end_pos+1],para[target_batch.tolist()[0][0][0]:target_batch.tolist()[0][1][0]+1])
        if score > 0.7 and goodExamples < 100:
            
            out += ('\n' + 'Score: ' + str(score) + '\n' + 'Question: ' + str(test_batch_quest_text) + '\n' + 'Context: ' + str(test_batch_para_text) + '\n' +
                'Predicted Answer\n' + str(getText(para[start_pos:end_pos+1]))) + '\n Target Answer: ' + str(getText(para[int(target_batch[0][0].item()): int(target_batch[0][1].item()) + 1])) + '\n'
            print ('\n')
            print ('Score', score)
            print ('Question', test_batch_quest_text)
            print ('Context', test_batch_para_text)
            print ('Predicted Answer', getText(para[start_pos:end_pos+1]))
            print ('Target Answer', getText(para[int(target_batch[0][0].item()): int(target_batch[0][1].item()) + 1]))
            print ('\n')
            goodExamples += 1
        
        if score < 0.4 and badExamples < 100:
            print ('\n')
            print ('Score', score)
            print ('Question', test_batch_quest_text)
            print ('Context', test_batch_para_text)
            print ('Predicted Answer', getText(para[start_pos:end_pos+1]))
            print ('Target Answer', getText(para[int(target_batch[0][0].item()): int(target_batch[0][1].item()) + 1]))
            print ('\n')
            out += ('\n' + 'Score: ' + str(score) + '\n' + 'Question: ' + str(test_batch_quest_text) + '\n' + 'Context: ' + str(test_batch_para_text) + '\n' +
                'Predicted Answer\n' + str(getText(para[start_pos:end_pos+1]))) + '\n Target Answer: ' + str(getText(para[int(target_batch[0][0].item()): int(target_batch[0][1].item()) + 1])) + '\n'
            badExamples += 1

        total_f1 += score
        # if i_batch % 100 == 0:
        #     print(total_f1/(i_batch+1))
    resultFile.write(out)
    print(f"Final Average F1 score (across {len(iterator)} examples): {total_f1/len(iterator)}")


# class dataset(Dataset):
#     def __init__(self, data_dir):
#         super(dataset, self).__init__()
#         self.df = data_dir
                
#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, i): # return single data item
#         dict_ret = {}
        
#         #print ('Index', i, self.df['Answer'][0])
#         i += valStartIndex

#         answerWindow = [int(self.df['Answer'][i][0][0]), int(self.df['Answer'][i][0][1]) - 1]
#         lst_quest, _, _ = vec_int(self.df['Question'][i], count_missing)
#         lst_context, _, _ = vec_int(self.df['Context'][i], count_missing)
        
#         dict_ret['Question_Txt'] = self.df['Question'][i]
#         dict_ret['Question_Tensor'] = torch.LongTensor(lst_quest)
#         dict_ret['Context_Txt'] = self.df['Context'][i]
#         dict_ret['Context_Tensor'] = torch.LongTensor(lst_context)
#         dict_ret['Answer'] = torch.LongTensor(answerWindow)
        
#         return dict_ret
    
# df_format_full = pd.read_pickle("processed_data.pkl")
# df_format_final = df_format_full.iloc[0:valStartIndex, :]
# val_df = df_format_full.iloc[valStartIndex:valExamples, :]

# print (len(val_df))

# print ('Len', len(df_format_full), len(word2idx), len(df_format_final))
    
# #train_data = dataset(df_format_final)
# val_data = dataset(val_df)


# # create train and test dataloader objects
# #train_loader = torch.utils.data.DataLoader(train_data, batch_size = 1, shuffle = True) 
# val = torch.utils.data.DataLoader(val_data, batch_size = 1, shuffle = True) 

# # uncomment below for testing
# for index, (df) in enumerate(val):
#     questionText = df['Question_Txt']
#     questionTensor = df["Question_Tensor"]
#     contextText = df["Context_Txt"]
#     contextTensor = df['Context_Tensor']
#     answer = df['Answer']
#     #print (df)
#     #break
# print ('Done')

val_loop_s2s(model,val_loader)