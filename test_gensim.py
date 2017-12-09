import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora

# # remove common words and tokenize
# stoplist = set('for a of the and to in'.split())
stoplist = set(line.strip().lower() for line in open('../TextMiningProj/StopWords/ND_Stop_Words_Generic.txt'))
print(stoplist)
# texts = [[word for word in document.lower().split() if word not in stoplist]
#           for document in documents]

# # remove words that appear only once
# from collections import defaultdict
# frequency = defaultdict(int)
# for text in texts:
#      for token in text:
#          frequency[token] += 1

# texts = [[token for token in text if frequency[token] > 1]
#           for text in texts]

# from pprint import pprint  # pretty-printer
# pprint(texts)

# dictionary = corpora.Dictionary(line.lower().split() for line in open('../TextMiningProj/Data/Moby_Dick.txt'))
# # dictionary = corpora.Dictionary(texts)
# dictionary.save('moby_dick.dict')  # store the dictionary, for future reference
# print(dictionary)

# print(dictionary.token2id)

# new_doc = "Human computer interaction"
# new_vec = dictionary.doc2bow(new_doc.lower().split())
# print(new_vec)

# corpus = [dictionary.doc2bow(text) for text in texts]
# corpora.MmCorpus.serialize('moby_dick.mm', corpus)  # store to disk, for later use
# print(corpus)

class MyCorpus(object):
    def __iter__(self):
        for line in open('../TextMiningProj/Data/Moby_Dick.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())


corpus = MyCorpus()  # doesn't load the corpus into memory!
# print(corpus_memory_friendly)
#
# for vector in corpus_memory_friendly:  # load one vector into memory at a time
#     print(vector)
#

from six import iteritems
# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('../TextMiningProj/Data/Moby_Dick.txt'))
# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
dictionary.save('moby_dick.dict')


# list(dictionary.keys())
# for k,v in dictionary.items():
#     print(k,v)


corpora.MmCorpus.serialize('moby_dick.mm', corpus)
corpora.BleiCorpus.serialize('moby_dick.lda-c', corpus)

# print(corpus)