import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from preprocessor import Preprocessor
import tomotopy as tp
from gensim import corpora, models

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
                p_z = (self.n_zt[:, id] + self.beta) * (self.n_mz[m, :] + self.alpha) / (self.n_z + self.vocab_size * self.beta)
                z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                z_n.append(z)
                self.n_mz[m, z] += 1
                self.n_zt[z, id] += 1
                self.n_z[z] += 1
            self.z_mn.append(np.array(z_n))

    def compute_pz(self, m, id, z):
        return ((self.n_zt[z, id] + self.beta) / (self.n_z[z] + self.vocab_size * self.beta)) * \
               ((self.n_mz[m, z] + self.alpha) / (len(self.corpus[m]) + self.num_topics * self.alpha))

    def inference(self):
        for m, doc in enumerate(self.corpus):
            for n, (id, _) in enumerate(doc):
                z = self.z_mn[m][n]
                self.n_mz[m, z] -= 1
                self.n_zt[z, id] -= 1
                self.n_z[z] -= 1

                p_z = np.array([self.compute_pz(m, id, zi) for zi in range(self.num_topics)])
                z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

                self.z_mn[m][n] = z
                self.n_mz[m, z] += 1
                self.n_zt[z, id] += 1
                self.n_z[z] += 1

    def get_topic_terms(self, topic, topn=10):
        topic = self.n_zt[topic, :] / self.n_zt[topic, :].sum()
        topn_ids = np.argsort(topic)[:-topn-1:-1]
        return [(self.dictionary[id], topic[id]) for id in topn_ids]
    

    def train(self, iterations):
        print('Training LDA...')
        for i in range(iterations):  # number of iterations
            self.inference()

    def results(self, iterations=100):

        self.train(iterations)
        # Print out results
        for i in range(self.num_topics):
            terms = self.get_topic_terms(i)
            topic = ' '.join([term for term, probability in terms])
            print(f'Topic {i}: {topic}')

            # print(' '.join([term for term, probability in terms]))
            # for term, probability in terms:
            #     print(f'{term}: {probability}')

    def get_perplexity(self):
        total_log_likelihood = 0
        total_num_words = 0
        for document in self.corpus:
            for id, freq in document:
                total_log_likelihood += np.log((self.n_zt[:, id] + self.beta) / (self.n_z + self.vocab_size * self.beta)) * freq
                total_num_words += freq
        return np.exp(-total_log_likelihood / total_num_words)
    
    def get_top_words(self, topic, topn=10):
        topic = self.n_zt[topic, :] / self.n_zt[topic, :].sum()
        topn_ids = np.argsort(topic)[:-topn-1:-1]
        return [self.dictionary[id] for id in topn_ids]

    def get_coherence(self, topn=10):
        topics = [self.get_top_words(i, topn) for i in range(self.num_topics)]
        cm = gensim.models.CoherenceModel(topics=topics, dictionary=self.dictionary, corpus=self.corpus, coherence='u_mass')
        return cm.get_coherence()

    def get_topic_diversity(self, topn=10):
        unique_words = set()
        total_words = 0
        for i in range(self.num_topics):
            top_words = self.get_top_words(i, topn)
            unique_words.update(top_words)
            total_words += len(top_words)
        return len(unique_words) / total_words