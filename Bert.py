#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(u'pip install pytorch-pretrained-bert')


# In[2]:


from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[ ]:


from keras.preprocessing.sequence import pad_sequences

data = ['In what country is Normandy located?', 'Today is a good day']
sentences = ["[CLS] " + d + " [SEP]" for d in data]
print (sentences)
tokenized_texts = [tokenizer.tokenize(sentence) for sentence in sentences]
print (tokenized_texts)

MAX_LEN = 9
indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in tokenized_texts]
indexed_tokens
input_ids = pad_sequences(indexed_tokens, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
input_ids


# In[23]:





# In[24]:


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


# In[25]:


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
    


# In[86]:


def getQuestionBertEmbeddings(sentence):
    sentences = "[CLS]" + sentence + "[SEP]"
    tokenized_text = tokenizer.tokenize(sentence)
    indexed_token = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_token])
    segments_tensors = torch.tensor([segments_ids])
    print (segments_tensors, tokens_tensor, len(tokenized_text))
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    sentence_embedding = torch.mean(encoded_layers[11], 1)
    return sentence_embedding

def getContextBertEmbeddings(sentence):
    sentences = "[CLS]" + sentence + "[SEP]"
    tokenized_text = tokenizer.tokenize(sentence)
    indexed_token = tokenizer.convert_tokens_to_ids(tokenized_text)
    batch_i = 0
    print (indexed_token)
    # Convert inputs to PyTorch tensors
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_token])
    segments_tensors = torch.tensor([segments_ids])
    print (segments_tensors, tokens_tensor, len(tokenized_text))
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    token_embeddings = [] 

# For each token in the sentence...
    for token_i in range(len(tokenized_text)):
      # Holds 12 layers of hidden states for each token 
      hidden_layers = [] 
      # For each of the 12 layers...
      for layer_i in range(len(encoded_layers)):
        # Lookup the vector for `token_i` in `layer_i`
        vec = encoded_layers[layer_i][batch_i][token_i]
        hidden_layers.append(vec)
      token_embeddings.append(hidden_layers)

    # Sanity check the dimensions:
    print ("Number of tokens in sequence:", len(token_embeddings))
    print ("Number of layers per token:", len(token_embeddings[0]))

    concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] # [number_of_tokens, 3072]
    summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] # [number_of_tokens, 768]
    print (torch.stack(summed_last_4_layers).shape)
    return torch.stack(summed_last_4_layers)


# In[89]:


class dataset(Dataset):
    def __init__(self, data_dir):
        super(dataset, self).__init__()
        self.df = data_dir
                
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i): # return single data item

        answerWindow = [int(df_format['Answer'][i][0]), int(df_format['Answer'][i][1])]
        bertQuestion = getQuestionBertEmbeddings(df_format['Question'][i])
        bertContext = getContextBertEmbeddings(df_format['Context'][i])
        print ('Question Shape', bertQuestion.shape)
        print ('Context Shape', bertContext.shape)
        return torch.LongTensor(vec_int(df_format['Question'][i])), torch.LongTensor(vec_int(df_format['Context'][i])), torch.LongTensor(answerWindow)
    
train_data = dataset(df_format)
# test_data = dataset(df_format)


# create train and test dataloader objects
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 1, shuffle = True) 
#test_loader = torch.utils.data.DataLoader(test_data, batch_size = bs, collate_fn = collate, shuffle = False) 


# In[90]:


for index, (df) in enumerate(train_loader):
    question = df[0]
    context = df[1]
    answer = df[2]
    break


# In[ ]:




