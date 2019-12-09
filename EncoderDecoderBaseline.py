#!/usr/bin/env python3

import torch
import torch.nn as nn


class BasicS2S(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, bidir=False):
        """
        Initiate Pointer-Net

        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param bool bidir: Bidirectional
        """
        super(BasicS2S, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.question_encoder = nn.LSTM(embedding_dim, hidden_dim)
        self.para_encoder = nn.LSTM(embedding_dim, hidden_dim)
        self.decoder = nn.LSTM(2*hidden_dim,hidden_dim)
        # 0 - neither, 1 - start token, 2 - end token
        self.lin_op = nn.Linear(hidden_dim,3)

    def forward(self, inputs, questions):
        """
        PointerNet - Forward-pass

        :param Tensor inputs: Input sequence (paragraph)
        :param Tensor questions: Questions sequence
        :return: Pointers probabilities and indices
        """

        batch_size = inputs.size(0)
        input_length = inputs.size(1)
        quest_length = questions.size(1)

        # input_len * 1
        inputs = inputs.view(batch_size * input_length, -1)
        # ques_len * 1
        quest_inputs = questions.view(batch_size * quest_length, -1)
        # inp_len * 1 * emb_dim
        embedded_para = self.embedding(inputs)
        # ques_len * 1 * emb_dim
        embedded_ques = self.embedding(quest_inputs)
        
        # ques_len * 1 * hidden, _
        question_lstm_out,_ = self.question_encoder.forward(embedded_ques)
        # para_len * 1 * hidden, _
        para_lstm_out, _ = self.para_encoder(embedded_para,(question_lstm_out[-1,:,:].unsqueeze(0),question_lstm_out[-1,:,:].unsqueeze(0))) 

        # broadcast and then concatenate question hidden state to each para hidden output
        # 1 * 1 * hidden
        question_final_lstm = question_lstm_out[-1,:,:].unsqueeze(0)
        question_bc = para_lstm_out.size(0) * [question_final_lstm]
        
        # para_len * 1 * hidden
        question_bc = torch.cat(question_bc,dim=0)
        # para_len * 1 * (2*hidden)
        para_quest_concat = torch.cat((para_lstm_out,question_bc),dim=2)

        # para_len * 1 * 3 -> para_len * 3
        final_op = self.lin_op(self.decoder(para_quest_concat)[0]).squeeze(dim=1)
        return  final_op