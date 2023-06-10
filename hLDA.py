import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from preprocessor import Preprocessor
import tomotopy as tp

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
                doc_as_tokens.extend([self.dictionary[id]] * freq)
            docs_as_tokens.append(doc_as_tokens)

        for doc_as_tokens in docs_as_tokens:
            hlda_model.add_doc(doc_as_tokens)

        for i in range(1):
            hlda_model.train(10)

        self.print_topics( topn=10)

    def get_perplexity(self):
        return self.model.perplexity

    def get_topic_diversity(self):
        topic_word_matrix = np.array([self.model.get_topic_word_dist(i) for i in range(self.model.k)])
        diversity_score = np.mean(np.apply_along_axis(lambda x: len(np.unique(x)), axis=1, arr=topic_word_matrix))
        return diversity_score
    
    def get_top_words(self, topic_id, topn=10):
        """This function returns the top words for a given topic."""
        topic_word_dist = self.model.get_topic_word_dist(topic_id)
        top_word_indices = np.argsort(topic_word_dist)[::-1][:topn]
        return [self.dictionary[idx] for idx in top_word_indices]
    
    def get_coherence(self, topn=10):
        """This function calculates and returns the coherence."""
        topics = [self.get_top_words(topic_id=i, topn=topn) for i in range(self.model.k)]
        cm = gensim.models.CoherenceModel(topics=topics, corpus=self.bow_corpus, dictionary=self.dictionary, coherence='u_mass')
        return cm.get_coherence()
    
    def print_topics(self, topn=10):
        for i in range(self.model.k):
            print('Topic', i, ':', self.get_top_words(i, topn=topn))