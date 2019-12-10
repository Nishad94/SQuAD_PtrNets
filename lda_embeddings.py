
# coding: utf-8

import numpy as np
import pandas as pd
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim import models
import re
import pickle
import nltk

df = pd.read_csv('glove.6B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
glove_dict = {key: val.values for key, val in df.T.items()}

# read SQuAD data
dev_set_csv = 'SQuAD-v1.1.csv'
data_csv = pd.read_csv(dev_set_csv, encoding = "ISO-8859-1")

# define stop words to be ignored
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.add('also')
stop_words.add('should')
stop_words.add('could')
stop_words.add('would')
stop_words.add('may')
stop_words.add('might')
stop_words.add('will')
stop_words.add('shall')
stop_words.add('many')
stop_words.add('however')
stop_words.add('therefore')
stop_words.add('hence')

# function to train lda model
def create_lda_model(data_csv, num_topics):
    custom_texts = []
    # for i in range(0, 30):
    for i in range(0, len(data_csv)):
        # add context vocab to dict
        context = data_csv['Context'][i]
        # hard-coded condition to train only contexts corresponding to first 20000 questions
        # if (context[: 42] == "Agricultural production is concentrated on"):
        #     break
        context = context.lower()
        context = context.replace("\'s", '')
        context = context.replace("\'", '')
        lst_words_context = re.findall(r"[\w']+|[.,!?;]", context)

        words = [w for w in lst_words_context if not w in stop_words] # remove stopwords
        words = [word for word in words if word.isalpha()] # remove punctuation
        custom_texts.append(words)

    custom_dict = Dictionary(custom_texts)
    custom_dict.filter_extremes(no_below=1, no_above=0.3)
    custom_corpus = [custom_dict.doc2bow(text) for text in custom_texts]

    # Train the model on the corpus.
    lda = models.LdaModel(custom_corpus, num_topics = num_topics, id2word = custom_dict)
    
    return lda, custom_dict

# function to create dict for topic embeddings
def get_topic_embeddings(lda, num_topics):
    dict_topic_embeddings = {}

    # get avg glove embeddings for top 50 words of each topic
    for i in range(0, num_topics):
        topics = lda.show_topic(topicid = i, topn = 50)
        lst_embeddings = []
        for j in range(0, len(topics)):
            word = topics[j][0]
            if (word in glove_dict):
                word_embedding = glove_dict[word]
                lst_embeddings.append(word_embedding)
        mean_embedding = np.mean(lst_embeddings, axis = 0)
        dict_topic_embeddings[i] = mean_embedding
        
    return dict_topic_embeddings

# function to preprocess context data
def preprocess(context):
    
    context = context.lower()
    context = context.replace("\'s", '')
    context = context.replace("\'", '')
    lst_words_context = re.findall(r"[\w']+|[.,!?;]", context)
    
    words = [w for w in lst_words_context if not w in stop_words] # remove stopwords
    words = [word for word in words if word.isalpha()] # remove punctuation
    
    return words

num_topics = 25


# create lda model
lda_model, dict_lda = create_lda_model(data_csv, num_topics)
# get dict of topic embeddings vectors
dict_topic_embeddings = get_topic_embeddings(lda_model, num_topics)

######################### Inference on a single context ##################################################
# obtain topic distribution for new context
context = data_csv['Context'][0]

# preprocess context
custom_texts = preprocess(context)
custom_texts = [d.split() for d in custom_texts]

custom_corpus = [dict_lda.doc2bow(text) for text in custom_texts]
dist_lda = lda_model[custom_corpus[0]]


# get weighted vector embeddings
vec_final = np.zeros((300))
for key in dict_topic_embeddings.keys():
    vec_temp = dist_lda[key][1]*dict_topic_embeddings[key]
    vec_final += vec_temp
