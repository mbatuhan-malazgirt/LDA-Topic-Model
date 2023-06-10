import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from preprocessor import Preprocessor
import tomotopy as tp


class correlatedLDA:
    def __init__(self, dictionary, bow_corpus):
        self.train_corpus = [[dictionary[id] for id, freq in doc] for doc in bow_corpus]
        self.dictionary = dictionary
        self.bow_corpus = bow_corpus


    def results(self, number_of_topics=1):
        # create an instance of the CTM model
        self.model = tp.CTModel(k=number_of_topics, min_cf=5)  # k is the number of topics

        # add the documents to the model
        for doc in self.train_corpus:
            self.model.add_doc(doc)
        # train the model
        for i in range(0, 100, 10):  # 100 is the number of iterations
            self.model.train(10)  # number of iterations in `train` function is 10

        # print out the topics
        for i in range(self.model.k):
            res = self.model.get_topic_words(i, top_n=10)
            topics = [word for word, _ in res]
            str = ''
            for word in topics:
                str = str + word + ' '

            print('Topic', i, ':', str)

    def get_perplexity(self):
        return self.model.perplexity    

    def get_topic_diversity(self):
        # Assuming `self.model` is the trained model and `self.dictionary` is the gensim Dictionary
        topic_word_matrix = np.array([self.model.get_topic_word_dist(i) for i in range(self.model.k)])
        diversity_score = np.mean(np.apply_along_axis(lambda x: len(np.unique(x)), axis=1, arr=topic_word_matrix))
        return diversity_score
    
    def get_top_words(self, topic_id, topn=10):
        topic = self.model.get_topic_words(topic_id=topic_id, top_n=topn)
        return [word for word, _ in topic]
    
    def get_coherence(self, topn=10):
        topics = [self.get_top_words(topic_id=i, topn=topn) for i in range(self.model.k)]
        cm = gensim.models.CoherenceModel(topics=topics, corpus=self.bow_corpus, dictionary=self.dictionary, coherence='u_mass')
        return cm.get_coherence()    