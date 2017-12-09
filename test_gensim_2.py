import os

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
if (os.path.exists("moby_dick.dict")):
    dictionary = corpora.Dictionary.load('moby_dick.dict')
    corpus = corpora.BleiCorpus('moby_dick.lda-c')
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")
#
# tfidf = models.TfidfModel(corpus)
#
# corpus_tfidf = tfidf[corpus]
# # for doc in corpus_tfidf:
# #      print(doc)
#
# lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10) # initialize an LSI transformation
# corpus_lsi = lsi[corpus_tfidf]
#
# lsi.print_topics(2)


model = models.LdaModel(corpus, id2word=dictionary, num_topics=90)
model.save('moby_dick.model')
print(model)