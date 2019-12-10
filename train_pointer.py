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
from Data_Generator import TSPDataset

from data_loader import train_loader, val_loader, word2idx
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
parser.add_argument('--bidir', default=False, action='store_true', help='Bidirectional')

params = parser.parse_args()

if params.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False

model = PointerNet(len(word2idx), params.embedding_size,
                   params.hiddens,
                   params.nof_lstms,
                   params.dropout,
                   params.bidir)

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

CCE = torch.nn.CrossEntropyLoss()
model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=params.lr)

losses = []

def test_loop(model,loader):
    model.eval()
    total_f1 = 0.0
    for i_batch, sample_batched in enumerate(loader):
        #iterator.set_description('Test Batch %i/%i' % (epoch+1, params.nof_epoch))
        test_batch_para = Variable(sample_batched["Context_Tensor"]).unsqueeze(2)
        test_batch_quest = Variable(sample_batched["Question_Tensor"]).unsqueeze(2)
        target_batch = Variable(sample_batched["Answer"]).unsqueeze(2)
        if USE_CUDA:
            test_batch_para = test_batch_para.cuda()
            test_batch_quest = test_batch_quest.cuda()
        o, p = model(test_batch_para,test_batch_quest)
        # Convert to list
        p_ = p.tolist()[0]
        p_.sort()
        para = test_batch_para.tolist()
        # Flatten 
        para = [l for item in para[0] for l in item]
        total_f1 += compute_f1(para[p_[0]:p_[1]+1],para[target_batch.tolist()[0][0][0]:target_batch.tolist()[0][1][0]+1])
        if i_batch % 100 == 0:
            print('Batch', i_batch, total_f1/(i_batch+1))
    import pdb
    pdb.set_trace()
    print(f"Final Average F1 score (across {len(loader)} examples): {total_f1/len(loader)}")

for epoch in range(params.nof_epoch):
    batch_loss = []
    iterator = tqdm(train_loader, unit='Batch')
    model.train()
    for i_batch, sample_batched in enumerate(iterator):
        iterator.set_description('Batch %i/%i' % (epoch+1, params.nof_epoch))

        train_batch_para = Variable(sample_batched["Context_Tensor"]).unsqueeze(2)
        train_batch_quest = Variable(sample_batched["Question_Tensor"]).unsqueeze(2)
        target_batch = Variable(sample_batched["Answer"]).unsqueeze(2)

        if USE_CUDA:
            train_batch_para = train_batch_para.cuda()
            train_batch_quest = train_batch_quest.cuda()
            target_batch = target_batch.cuda()
        #print (type(train_batch_para),type(train_batch_quest))
        o, p = model(train_batch_para,train_batch_quest)
        o = o.contiguous().view(-1, o.size()[-1])

        target_batch = target_batch.view(-1)
        loss = CCE(o, target_batch)

        losses.append(loss.item())
        batch_loss.append(loss.item())

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()

        iterator.set_postfix(loss='{}'.format(loss.item()))
        
    
    iterator.set_postfix(loss=np.average(batch_loss))
    torch.save(model.state_dict(), f"{time.time()}_{epoch}.pt")
    model.eval()
    test_loop(model,val_loader)
    
