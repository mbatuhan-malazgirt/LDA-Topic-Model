import random
import numpy as np

class LDAModel:
    def __init__(self, num_topics, docs, alpha=0.1, beta=0.1):
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.docs = docs

        self.vocab = list(set([word for doc in docs for word in doc]))
        self.vocab_size = len(self.vocab)
        self.word2id = {word: i for i, word in enumerate(self.vocab)}

        self.z_mn = []  # topics of words of documents
        self.n_mz = np.zeros((len(docs), num_topics)) + alpha  # doc-topic distribution
        self.n_zt = np.zeros((num_topics, self.vocab_size)) + beta  # topic-word distribution
        self.n_z = np.zeros(num_topics) + self.vocab_size * beta  # number of words each topic has

        for doc in docs:
            z_n = []
            for word in doc:
                p_z = self.n_z / sum(self.n_z)
                z = np.random.multinomial(1, p_z).argmax()
                z_n.append(z)
                self.n_mz[docs.index(doc), z] += 1
                self.n_zt[z, self.word2id[word]] += 1
                self.n_z[z] += 1
            self.z_mn.append(np.array(z_n))

    def inference(self):
        for m, doc in enumerate(self.docs):
            for n, word in enumerate(doc):
                z = self.z_mn[m][n]
                self.n_mz[m, z] -= 1
                self.n_zt[z, self.word2id[word]] -= 1
                self.n_z[z] -= 1

                p_z = (self.n_zt[:, self.word2id[word]] / self.n_z) * (self.n_mz[m, :] / len(doc))
                z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

                self.z_mn[m][n] = z
                self.n_mz[m, z] += 1
                self.n_zt[z, self.word2id[word]] += 1
                self.n_z[z] += 1

    def get_topic_terms(self, topic, topn=10):
        topic = self.n_zt[topic, :] / self.n_zt[topic, :].sum()
        topn_ids = np.argsort(topic)[:-topn-1:-1]
        return [(id, topic[id]) for id in topn_ids]

docs = [
    ['apple', 'banana', 'apple', 'apple', 'banana', 'strawberry', 'strawberry', 'strawberry'],
    ['dog', 'dog', 'cat', 'dog', 'cat', 'cat', 'dog', 'cat'],
    ['car', 'bike', 'car', 'bike', 'car', 'car', 'bike', 'bike'],
    # Add more documents here
]

lda = LDAModel(2, docs)

for i in range(100):  # number of iterations
    lda.inference()

# Print out results
for i in range(lda.num_topics):
    print(f'Topic {i}:')
    terms = lda.get_topic_terms(i)
    for term, probability in terms:
        print(f'{lda.vocab[term]}: {probability}')
