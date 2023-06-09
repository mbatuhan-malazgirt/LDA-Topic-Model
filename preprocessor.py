import re
import html

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

