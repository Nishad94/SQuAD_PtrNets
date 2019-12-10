import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import argparse
from tqdm import tqdm

from data_loader import train_loader, word2idx

iterator = tqdm(train_loader, unit='Batch')
for i_batch, sample_batched in enumerate(iterator):

    train_batch_para = Variable(sample_batched["Context_Tensor"]).unsqueeze(2).long()
    train_batch_quest = Variable(sample_batched["Question_Tensor"]).unsqueeze(2).long()
    target_batch = Variable(sample_batched["Answer"]).unsqueeze(2)
    train_batch_quest_text = sample_batched["Question_Txt"]
    train_batch_para_text = sample_batched["Context_Txt"] 
    target_batch = target_batch.view(-1)
    import pdb
    pdb.set_trace()