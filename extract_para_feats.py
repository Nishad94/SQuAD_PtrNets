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

iterator = tqdm(train_loader, unit='Batch')

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

context_feats = []
for i_batch, sample_batched in enumerate(iterator):
    train_batch_quest_text = sample_batched["Question_Txt"]
    emb = getQuestionBertEmbeddings(train_batch_quest_text[0])
    emb = emb.squeeze(0)
    emb = emb.unsqueeze(1).unsqueeze(2)
    context_feats.append(emb)

concat_feats = torch.stack(context_feats)
import numpy as np

npy_frames = concat_feats.cpu().numpy()
np.save("ques_feats_train.npy",npy_frames)