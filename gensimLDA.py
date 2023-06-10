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
            # print('Topic: {} \nWords: {}'.format(idx, topic))

            prob_words = topic.split('+')
            words = [word.split('*')[1].replace('"', '').strip()for word in prob_words]
            str = ''
            for word in words:
                str = str + word + ' '
            print('Topic: {}: {}'.format(idx, str))

    def get_perplexity(self):
        return self.model.log_perplexity(self.corpus)

    def get_coherence(self):
        coherence_model = gensim.models.CoherenceModel(model=self.model, corpus=self.corpus, coherence='u_mass')
        return coherence_model.get_coherence()
    
    def get_top_words(self, topic_id, topn=10):
        topic = self.model.get_topic_terms(topicid=topic_id, topn=topn)
        return [self.id2word[id] for id, _ in topic]

    def get_topic_diversity(self, topn=10):
        """This function calculates and returns the topic diversity."""
        unique_words = set()
        total_words = 0
        for i in range(self.num_topics):
            top_words = self.get_top_words(i, topn)
            unique_words.update(top_words)
            total_words += len(top_words)
        return len(unique_words) / total_words