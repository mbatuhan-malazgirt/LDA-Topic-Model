import pandas as pd
import numpy as np
from preprocessor import Preprocessor
from lda import LDAModel
from gensimLDA import gensimLDA
from correlatedLDA import correlatedLDA
from hLDA import hLDA

preprocessor = Preprocessor('abcnews-date-text.csv')
preprocessor.readDocuments()
preprocessor.preprocessDocuments()
preprocessor.bagofwords()
preprocessor.tfidf()

 # k is the number of topics, dictionary is the dictionary, bow_corpus is the bag of words corpus
LDAModel = LDAModel(1, preprocessor.dictionary, preprocessor.bow_corpus)   
LDAModel.results()

gensimLDA = gensimLDA(preprocessor.bow_corpus, 1, preprocessor.dictionary, 2)
gensimLDA.results()

correlatedLDA = correlatedLDA(preprocessor.dictionary, preprocessor.bow_corpus)
correlatedLDA.results()

hLDAModel = hLDA(preprocessor.bow_corpus, preprocessor.dictionary)
hLDAModel.results()