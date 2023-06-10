import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import nltk
nltk.download('wordnet')

class Preprocessor:

    def __init__(self, directory):
        self.directory = directory

    def readDocuments(self):
        data = pd.read_csv(self.directory, error_bad_lines=False, nrows=1000);
        data_text = data[['headline_text']]
        data_text['index'] = data_text.index
        self.documents = data_text

    def lemmatize_stemming(self, text):
        stemmer = nltk.stem.PorterStemmer()
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    
    def preprocess(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))
        return result
    
    def preprocessDocuments(self):
        self.processed_docs = self.documents['headline_text'].map(self.preprocess)

    def bagofwords(self):
        self.dictionary = gensim.corpora.Dictionary(self.processed_docs)
        self.dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.processed_docs]

    def tfidf(self):
        tfidf = models.TfidfModel(self.bow_corpus)
        self.corpus_tfidf = tfidf[self.bow_corpus]

    def preprocessAndSave(self):
        processed_data = pd.read_csv(self.directory, error_bad_lines=False);
        # processed_data = pd.read_csv(self.directory, error_bad_lines=False, nrows=100);
        processed_data['headline_text'] = processed_data['headline_text'].map(self.preprocess)
        processed_data.to_csv('processed-abcnews.csv', index=False)