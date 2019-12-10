#!/usr/bin/env python3

import torch
import torch.nn as nn



if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
modelBERT = BertModel.from_pretrained('bert-base-uncased')
if USE_CUDA:
    modelBERT.cuda()
modelBERT.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def getContextBertEmbeddings(sentence):
    sentences = "[CLS]" + sentence + "[SEP]"
    tokenized_text = tokenizer.tokenize(sentence)
    if len(tokenized_text) > 511:
        tokenized_text = tokenized_text[:511]
    
    indexed_token = tokenizer.convert_tokens_to_ids(tokenized_text)
    batch_i = 0
    #print (indexed_token)
    # Convert inputs to PyTorch tensors
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_token])

    segments_tensors = torch.tensor([segments_ids])

    if USE_CUDA:
        tokens_tensor = tokens_tensor.cuda()
        segments_tensors = segments_tensors.cuda()

    #print (segments_tensors, tokens_tensor, len(tokenized_text))
    with torch.no_grad():
        encoded_layers, _ = modelBERT(tokens_tensor, segments_tensors)
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
    # print ("Number of tokens in sequence:", len(token_embeddings))
    # print ("Number of layers per token:", len(token_embeddings[0]))

    concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] # [number_of_tokens, 3072]
    summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] # [number_of_tokens, 768]
    #print (torch.stack(summed_last_4_layers).shape)
    return torch.stack(summed_last_4_layers)



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

    def forward(self, inputs, questions, inputs_text, questions_text):
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


        # embedded_para = self.embedding(inputs)
        # # ques_len * 1 * emb_dim
        # embedded_ques = self.embedding(quest_inputs)




        embedded_para = getContextBertEmbeddings(inputs_text[0])
        embedded_ques = getContextBertEmbeddings(questions_text[0])

        
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