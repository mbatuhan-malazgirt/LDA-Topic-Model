import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from preprocessor import Preprocessor
import tomotopy as tp


class gensimLDA:
    def __init__(self, corpus, num_topics, id2word, passes):
        self.corpus = corpus
        self.num_topics = num_topics
        self.id2word = id2word
        self.passes = passes
        self.model = gensim.models.LdaModel(self.corpus, num_topics = self.num_topics, id2word = self.id2word, passes = self.passes)

    def results(self):
        for idx, topic in self.model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))