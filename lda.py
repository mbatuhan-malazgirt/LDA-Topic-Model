import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from preprocessor import Preprocessor
import tomotopy as tp

class LDAModel:
    def __init__(self, num_topics, dictionary, corpus, alpha=0.1, beta=0.1):
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.dictionary = dictionary
        self.corpus = corpus

        self.vocab_size = len(self.dictionary)
        self.z_mn = []  # topics of words of documents
        self.n_mz = np.zeros((len(corpus), num_topics)) + alpha  # doc-topic distribution
        self.n_zt = np.zeros((num_topics, self.vocab_size)) + beta  # topic-word distribution
        self.n_z = np.zeros(num_topics) + self.vocab_size * beta  # number of words each topic has

        for m, doc in enumerate(corpus):
            z_n = []
            for id, _ in doc:
                p_z = self.n_z / sum(self.n_z)
                z = np.random.multinomial(1, p_z).argmax()
                z_n.append(z)
                self.n_mz[m, z] += 1
                self.n_zt[z, id] += 1
                self.n_z[z] += 1
            self.z_mn.append(np.array(z_n))

    def inference(self):
        for m, doc in enumerate(self.corpus):
            for n, (id, _) in enumerate(doc):
                z = self.z_mn[m][n]
                self.n_mz[m, z] -= 1
                self.n_zt[z, id] -= 1
                self.n_z[z] -= 1

                p_z = abs((self.n_zt[:, id] / self.n_z) * (self.n_mz[m, :] / max(len(doc), 1e-10)))
                z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

                self.z_mn[m][n] = z
                self.n_mz[m, z] += 1
                self.n_zt[z, id] += 1
                self.n_z[z] += 1

    def get_topic_terms(self, topic, topn=10):
        topic = self.n_zt[topic, :] / self.n_zt[topic, :].sum()
        topn_ids = np.argsort(topic)[:-topn-1:-1]
        return [(self.dictionary[id], topic[id]) for id in topn_ids]



preprocessor = Preprocessor('abcnews-date-text.csv')
preprocessor.readDocuments()
preprocessor.preprocessDocuments()
preprocessor.bagofwords()
preprocessor.tfidf()

docs = preprocessor.bow_corpus

lda = LDAModel(1, preprocessor.dictionary, preprocessor.bow_corpus)

print('Training LDA...')
for i in range(2):  # number of iterations
    lda.inference()

# Print out results
for i in range(lda.num_topics):
    print(f'Topic {i}:')
    terms = lda.get_topic_terms(i)
    for term, probability in terms:
        print(f'{term}: {probability}')