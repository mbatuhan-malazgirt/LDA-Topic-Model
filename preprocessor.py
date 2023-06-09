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
        data = pd.read_csv(self.directory, error_bad_lines=False);
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
        
'''
preprocessor = Preprocessor('abcnews-date-text.csv')
preprocessor.readDocuments()
preprocessor.preprocessDocuments()
preprocessor.bagofwords()
preprocessor.tfidf()

from pprint import pprint
for doc in preprocessor.corpus_tfidf:
    pprint(doc)
    break
'''

'''
doc_sample = preprocessor.documents[preprocessor.documents['index'] == 4310].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocessor.preprocess(doc_sample))
'''

'''
class Preprocessor:
    def __init__(self, stopword_file):
        self.stopwords = self.load_stopwords(stopword_file)

    def load_file(self, filename):
        with open(filename, 'r', encoding='latin-1') as file:
            data = file.read()
            data = html.unescape(data)  # unescape html
        return data

    def load_stopwords(self, stopword_file):
        with open(stopword_file, 'r', encoding='latin-1') as file:
            stopwords = file.readlines()
            stopwords = [word.strip() for word in stopwords]
        return set(stopwords)

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())  # return list of tokens

    def remove_stopwords(self, words):
        cleaned_words = [word.lower() for word in words if word.lower() not in self.stopwords]
        return cleaned_words

    def extract_text(self, data):
        text_pattern = r'<REUTERS(.*?)</REUTERS>'  # match text between <REUTERS> tags
        title_pattern = r'<TITLE>(.*?)</TITLE>'  # match text between <TITLE> tags
        body_pattern = r'<BODY>(.*?)</BODY>'  # match text between <BODY> tags
        text_matches = re.findall(text_pattern, data, re.DOTALL | re.IGNORECASE)  # find all matches which extracts the articles

        articles = []
        for text in text_matches:
            title_match = re.search(title_pattern, text, re.DOTALL | re.IGNORECASE)  # find title
            body_match = re.search(body_pattern, text, re.DOTALL | re.IGNORECASE)  # find body
            title = title_match.group(1) if title_match else ''  # get title text
            body = body_match.group(1) if body_match else ''  # get body text
            lewis_split = re.search(r'LEWISSPLIT="(.*?)"', text, re.IGNORECASE)  # get lewis split
            topic_bool = re.search(r'TOPICS="(.*?)"', text, re.IGNORECASE).group(1)  # get whether topics exist or not
            topics = []
            if topic_bool == 'YES':  # if topics exist
                topic_matches = re.findall(r'<TOPICS>(.*?)</TOPICS>', text, re.DOTALL | re.IGNORECASE)  # find all matches which extracts the topics
                for topic_set in topic_matches:  # for each topic set
                    topic_list = re.findall(r'<D>(.*?)</D>', topic_set, re.DOTALL | re.IGNORECASE)  # find all topics
                    topics.extend(topic_list)
                text_tokens = self.tokenize(title + ' ' + body)  # tokenize title and body
                text_tokens = self.remove_stopwords(text_tokens)  # remove stopwords from title and body
            articles.append((text_tokens, topics, lewis_split.group(1) if lewis_split else None))  # append tuple of text, topics, and lewis split as article
        return articles

    def preprocess(self, directory):
        data = ''
        for i in range(1):
            name = 'reut2-0' + str(i).zfill(2) + '.sgm'
            data += self.load_file(directory + name)

        articles = self.extract_text(data)
        return articles

"""# Usage example
stopword_file = 'stopwords.txt'  # stopwords file
directory = 'reuters21578/'  # directory containing the data files

preprocessor = Preprocessor(stopword_file)
articles = preprocessor.preprocess(directory)"""
'''
