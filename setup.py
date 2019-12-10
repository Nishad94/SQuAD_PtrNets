import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import argparse
from tqdm import tqdm

from data_loader import train_loader, val_loader, word2idx

# iterator = tqdm(train_loader, unit='Batch')

# question_texts = ""
# q_text_list = []
# para_texts = ""
# spans = ""
# targets_long = ""
# idx2word = {}
# for k in word2idx:
#     idx2word[word2idx[k]] = k
# for i_batch, sample_batched in enumerate(iterator):
#     train_batch_para = Variable(sample_batched["Context_Tensor"]).unsqueeze(2).long()
#     train_batch_quest = Variable(sample_batched["Question_Tensor"]).unsqueeze(2).long()
#     target_batch = Variable(sample_batched["Answer"]).unsqueeze(2)
#     train_batch_quest_text = sample_batched["Question_Txt"]
#     train_batch_para_text = sample_batched["Context_Txt"] 
#     target_batch = target_batch.view(-1).tolist()
#     q_text_list.append(train_batch_quest_text[0])
#     question_texts += train_batch_quest_text[0] + "\n"
#     para_texts += train_batch_para_text[0].replace("\n","") + "\n"
#     spans += str(target_batch[0]) + " " + str(target_batch[1]) + "\n"
#     # targets = torch.zeros(o.size(0))
#     # targets[target_batch[0]] = 1
#     # targets[target_batch[1]] = 2



# with open("ques_train_short.txt",'w') as f:
#     f.write(question_texts[:-1])
# with open("para_train_short.txt","w") as f:
#     f.write(para_texts[:-1])
# with open("span_train_short.txt","w") as f:
#     f.write(spans[:-1])


iterator = tqdm(val_loader, unit='Batch')

question_texts = ""
para_texts = ""
spans = ""
for i_batch, sample_batched in enumerate(iterator):
    train_batch_para = Variable(sample_batched["Context_Tensor"]).unsqueeze(2).long()
    train_batch_quest = Variable(sample_batched["Question_Tensor"]).unsqueeze(2).long()
    target_batch = Variable(sample_batched["Answer"]).unsqueeze(2)
    train_batch_quest_text = sample_batched["Question_Txt"]
    train_batch_para_text = sample_batched["Context_Txt"] 
    target_batch = target_batch.view(-1).tolist()
    question_texts += train_batch_quest_text[0] + "\n"
    para_texts += train_batch_para_text[0].replace("\n","") + "\n"
    spans += str(target_batch[0]) + " " + str(target_batch[1]) + "\n"

with open("ques_val_short.txt",'w') as f:
    f.write(question_texts[:-1])
with open("para_val_short.txt","w") as f:
    f.write(para_texts[:-1])
with open("span_val_short.txt","w") as f:
    f.write(spans[:-1])