import logging
from gensim import corpora, models, similarities
import os
from pprint import pprint
import json
import numpy as np
import warnings
import pyLDAvis
warnings.filterwarnings('ignore')
import gensim
import pyLDAvis.gensim

dictionary = gensim.corpora.Dictionary.load('moby_dick.dict')
corpus = gensim.corpora.MmCorpus('moby_dick.mm')
lda = gensim.models.ldamodel.LdaModel.load('moby_dick.model')


print(dictionary)
print(corpus)
print(lda)

# print(lda.print_topics(num_topics=2, num_words=4))

# for i in lda.print_topics():
#     for j in i: print(j)

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda, corpus, dictionary)
