import pandas as pd
import numpy as np
from preprocessor import Preprocessor
from lda import LDAModel
from gensimLDA import gensimLDA
from correlatedLDA import correlatedLDA
from hLDA import hLDA

np.random.seed(0)

preprocessor = Preprocessor('abcnews-date-text.csv')
preprocessor.readDocuments()
preprocessor.preprocessDocuments()
preprocessor.bagofwords()
preprocessor.tfidf()

print()
print("LDA Model")
print()

 # k is the number of topics, dictionary is the dictionary, bow_corpus is the bag of words corpus
LDAModel = LDAModel(1, preprocessor.dictionary, preprocessor.bow_corpus)   
LDAModel.results(10)
LDAModelPerplexity = LDAModel.get_perplexity()
LDAModelCoherence = LDAModel.get_coherence()
LDAModelTopicdiv = LDAModel.get_topic_diversity()
print("Perplexity: " + str(np.mean(LDAModelPerplexity)))
print("Coherence: " + str(LDAModelCoherence))
print("Topic Diversity: " + str(LDAModelTopicdiv))




print()
print("Gensim LDA")
print()

gensimLDA = gensimLDA(preprocessor.bow_corpus, 1, preprocessor.dictionary, 1)
gensimLDA.results()
gensimLDAPerplexity = gensimLDA.get_perplexity()
gensimLDAcoherence = gensimLDA.get_coherence()
gensimLDAcoherence = gensimLDA.get_topic_diversity()
print("Perplexity: " + str(gensimLDAPerplexity))
print("Coherence: " + str(gensimLDAcoherence))
print("Topic Diversity: " + str(gensimLDAcoherence))




print()
print("Correlated LDA")
print()

correlatedLDA = correlatedLDA(preprocessor.dictionary, preprocessor.bow_corpus)
correlatedLDA.results(1)
correlatedLDAPerplexity = correlatedLDA.get_perplexity()
correlatedLDAtopicdiv = correlatedLDA.get_topic_diversity()
correlatedLDAtopicdiv = correlatedLDA.get_coherence()
print("Perplexity: " + str(correlatedLDAPerplexity))
print("Coherence: " + str(correlatedLDAtopicdiv))
print("Topic Diversity: " + str(correlatedLDAtopicdiv))


print()
print("hLDA")
print()

hLDAModel = hLDA(preprocessor.bow_corpus, preprocessor.dictionary)
hLDAModel.results()
hLDAModelPerplexity = hLDAModel.get_perplexity()
hLDAModelCoherence = hLDAModel.get_coherence()
hLDAModeltopicdiv = hLDAModel.get_topic_diversity()
print("Perplexity: " + str(hLDAModelPerplexity))
print("Coherence: " + str(hLDAModelCoherence))
print("Topic Diversity: " + str(hLDAModeltopicdiv))
