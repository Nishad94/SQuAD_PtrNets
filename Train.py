#!/usr/bin/env python3

"""
Pytorch implementation of Pointer Network.

http://arxiv.org/pdf/1506.03134v1.pdf.
"""

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




parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# Data
parser.add_argument('--train_size', default=1000000, type=int, help='Training data size')
parser.add_argument('--val_size', default=10000, type=int, help='Validation data size')
parser.add_argument('--test_size', default=10000, type=int, help='Test data size')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
# Train
parser.add_argument('--nof_epoch', default=50000, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# GPU
parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
# TSP
parser.add_argument('--nof_points', type=int, default=5, help='Number of points in TSP')
# Network
parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
parser.add_argument('--nof_lstms', type=int, default=2, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')

params = parser.parse_args()

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

if params.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False

# from sentence_transformers import SentenceTransformer
# modelBERT = SentenceTransformer('bert-base-nli-mean-tokens')




# model = PointerNet(params.embedding_size,
#                    params.hiddens,
#                    params.nof_lstms,
#                    params.dropout,
#                    params.bidir)
model = BasicS2S(len(word2idx),768,256)

# dataset = TSPDataset(params.train_size,
#                      params.nof_points)

# dataloader = DataLoader(dataset,
#                         batch_size=params.batch_size,
#                         shuffle=True,
#                         num_workers=4)


if USE_CUDA:
    model.cuda()
    net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
weight=torch.Tensor([1, 2, 1]).cuda()
CCE = torch.nn.CrossEntropyLoss(weight = weight)
model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=params.lr)

losses = []

def test_loop_s2s(model,loader):
    total_f1 = 0.0
    for i_batch, sample_batched in enumerate(loader):
        iterator.set_description('Test Batch %i/%i' % (epoch+1, params.nof_epoch))
        test_batch_para = Variable(sample_batched["Context_Tensor"]).unsqueeze(2)
        test_batch_quest = Variable(sample_batched["Question_Tensor"]).unsqueeze(2)
        target_batch = Variable(sample_batched["Answer"]).unsqueeze(2)
        test_batch_quest_text = sample_batched["Question_Txt"]
        test_batch_para_text = sample_batched["Context_Txt"]


        if USE_CUDA:
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
        total_f1 += compute_f1(para[start_pos:end_pos+1],para[target_batch.tolist()[0][0][0]:target_batch.tolist()[0][1][0]+1])
        if i_batch % 100 == 0:
            print(total_f1/(i_batch+1))
    print(f"Final Average F1 score (across {len(iterator)} examples): {total_f1/len(loader)}")

for epoch in range(params.nof_epoch):
    batch_loss = []
    iterator = tqdm(train_loader, unit='Batch')
    print ('Epoch', epoch)
    c = 0
    for i_batch, sample_batched in enumerate(iterator):
        iterator.set_description('Batch %i/%i' % (epoch+1, params.nof_epoch))

        train_batch_para = Variable(sample_batched["Context_Tensor"]).unsqueeze(2).long()
        train_batch_quest = Variable(sample_batched["Question_Tensor"]).unsqueeze(2).long()
        target_batch = Variable(sample_batched["Answer"]).unsqueeze(2)
        train_batch_quest_text = sample_batched["Question_Txt"]
        train_batch_para_text = sample_batched["Context_Txt"] 

        if USE_CUDA:
            train_batch_para = train_batch_para.cuda()
            train_batch_quest = train_batch_quest.cuda()
            target_batch = target_batch.cuda()

        target_batch = target_batch.view(-1)
        #print (target_batch)
        if target_batch[0] > 511 or target_batch[1] > 511:
            print (target_batch)
            continue
        # o, p = model(train_batch_para,train_batch_quest)
        # o = o.contiguous().view(-1, o.size()[-1])
        o = model(train_batch_para,train_batch_quest, train_batch_para_text, train_batch_quest_text)

        # print ('Here')
        # import pdb
        # pdb.set_trace()
        # Changes for baseline s2s

        targets = torch.zeros(o.size(0))
        targets[target_batch[0]] = 1
        targets[target_batch[1]] = 2
        if USE_CUDA:
            targets = targets.cuda()
        loss = CCE(o, targets.long())
        # end of changes

        #loss = CCE(o, target_batch)

        losses.append(loss.item())
        batch_loss.append(loss.item())

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()

        iterator.set_postfix(loss='{}'.format(loss.item()))


    iterator.set_postfix(loss=np.average(batch_loss))
    torch.save(model.state_dict(), f"{time.time()}_{epoch}.pt")
    test_loop_s2s(model,val_loader)
    
