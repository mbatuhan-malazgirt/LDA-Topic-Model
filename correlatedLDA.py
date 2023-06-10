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
        # prepare data for tomotopy
        self.train_corpus = [[dictionary[id] for id, freq in doc] for doc in bow_corpus]

    def results(self):
        # create an instance of the CTM model
        self.model = tp.CTModel(k=1, min_cf=5)  # k is the number of topics

        # add the documents to the model
        for doc in self.train_corpus:
            self.model.add_doc(doc)
        # train the model
        for i in range(0, 100, 10):  # 100 is the number of iterations
            self.model.train(10)  # number of iterations in `train` function is 10

        # print out the topics
        for i in range(self.model.k):
            res = self.model.get_topic_words(i, top_n=10)
            print('Topic', i, ':', [(word, prob) for word, prob in res])

    def get_perplexity(self):
        return self.model.perplexity    

    def get_topic_diversity(self):
        # Assuming `self.model` is the trained model and `self.dictionary` is the gensim Dictionary
        topic_word_matrix = np.array([self.model.get_topic_word_dist(i) for i in range(self.model.k)])
        diversity_score = np.mean(np.apply_along_axis(lambda x: len(np.unique(x)), axis=1, arr=topic_word_matrix))
        return diversity_score