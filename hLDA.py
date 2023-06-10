import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from preprocessor import Preprocessor
import tomotopy as tp

preprocessor = Preprocessor('abcnews-date-text.csv')
preprocessor.readDocuments()
preprocessor.preprocessDocuments()
preprocessor.bagofwords()
preprocessor.tfidf()


class hLDA:
    def __init__(self, bow_corpus, dictionary):
        self.bow_corpus = bow_corpus
        self.dictionary = dictionary

    def results(self):
        hlda_model = tp.HLDAModel(depth=2, # specifies the depth of the topic tree
                          alpha=0.1, # the higher => more topics documents can cover
                          gamma=1.0, # the higher => more topics can be created
                          eta=0.1, # the higher => topics become more diverse
                          seed=0)
        
        self.model = hlda_model
        
        docs = self.bow_corpus

        docs_as_tokens = []

        for doc in docs:
            doc_as_tokens = []
            for id, freq in doc:
                doc_as_tokens.extend([preprocessor.dictionary[id]] * freq)
            docs_as_tokens.append(doc_as_tokens)

        for doc_as_tokens in docs_as_tokens:
            hlda_model.add_doc(doc_as_tokens)

        for i in range(2):
            hlda_model.train(1)

        for k in range(hlda_model.k):
            print('Level: {} \tTopic: {}'.format(hlda_model.level(k), hlda_model.get_topic_words(k, top_n=5)))

    def get_perplexity(self):
        return self.model.perplexity

    def get_topic_diversity(self):
        topic_word_matrix = np.array([self.model.get_topic_word_dist(i) for i in range(self.model.k)])
        diversity_score = np.mean(np.apply_along_axis(lambda x: len(np.unique(x)), axis=1, arr=topic_word_matrix))
        return diversity_score