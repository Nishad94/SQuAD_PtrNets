
# coding: utf-8

# In[15]:

import numpy as np
import pandas as pd
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim import models
import re
import pickle
import nltk

# read SQuAD data
dev_set_csv = 'SQuAD-v1.1.csv'

data_csv = pd.read_csv(dev_set_csv, encoding = "ISO-8859-1")


# create dict of all words in all contexts
custom_texts = []
# for i in range(0, 30):
for i in range(0, len(data_csv)):
    # add context vocab to dict
    context = data_csv['Context'][i]
    # hard-coded condition to train only contexts corresponding to first 20000 questions
    if (context[: 42] == "Agricultural production is concentrated on"):
        break
    context = context.lower()
    context = context.replace("\'s", '')
    context = context.replace("\'", '')
    lst_words_context = re.findall(r"[\w']+|[.,!?;]", context)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [w for w in lst_words_context if not w in stop_words] # remove stopwords
    words = [word for word in words if word.isalpha()] # remove punctuation
    custom_texts.append(words)

custom_dict = Dictionary(custom_texts)
custom_corpus = [custom_dict.doc2bow(text) for text in custom_texts]


# Train the model on the corpus.
lda = models.ldamodel.LdaModel(custom_corpus, num_topics = 5, id2word = custom_dict)

# Save model to disk.
loc_file = 'lda_model.pkl'
lda.save(loc_file)

