import re
import html
import numpy as np

def load_file(filename):
    with open(filename, 'r', encoding='latin-1') as file:
        data = file.read()
        data = html.unescape(data) # unescape html
    return data

# load stopwords from file
def load_stopwords(stopword_file):
    with open(stopword_file, 'r', encoding='latin-1') as file:
        stopwords = file.readlines()
        stopwords = [word.strip() for word in stopwords]
    return set(stopwords)

# tokenize text
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower()) # return list of tokens

# remove stopwords from file
def remove_stopwords(words, stopwords):
    cleaned_words = [word.lower() for word in words if word.lower() not in stopwords] 
    return cleaned_words

def extract_text(data, stopwords): 
    text_pattern = r'<REUTERS(.*?)</REUTERS>' # match text between <REUTERS> tags
    title_pattern = r'<TITLE>(.*?)</TITLE>' # match text between <TITLE> tags
    body_pattern = r'<BODY>(.*?)</BODY>' # match text between <BODY> tags
    text_matches = re.findall(text_pattern, data, re.DOTALL | re.IGNORECASE) # find all matches which extracts the articles

    articles = []
    for text in text_matches:
        title_match = re.search(title_pattern, text, re.DOTALL | re.IGNORECASE) # find title
        body_match = re.search(body_pattern, text, re.DOTALL | re.IGNORECASE) # find body
        title = title_match.group(1) if title_match else '' # get title text
        body = body_match.group(1) if body_match else '' # get body text
        lewis_split = re.search(r'LEWISSPLIT="(.*?)"', text, re.IGNORECASE) # get lewis split
        topic_bool = re.search(r'TOPICS="(.*?)"', text, re.IGNORECASE).group(1) # get whether topics exist or not
        topics = []
        if(topic_bool == 'YES'): # if topics exist
            topic_matches = re.findall(r'<TOPICS>(.*?)</TOPICS>', text, re.DOTALL | re.IGNORECASE) # find all matches which extracts the topics
            for topic_set in topic_matches: # for each topic set
                topic_list = re.findall(r'<D>(.*?)</D>', topic_set, re.DOTALL | re.IGNORECASE) # find all topics
                topics.extend(topic_list)
            text_tokens = tokenize(title + ' ' + body) # tokenize title and body
            text_tokens = remove_stopwords(text_tokens, stopwords) # remove stopwords from title and body
        articles.append((text_tokens, topics, lewis_split.group(1) if lewis_split else None)) # append tuple of text, topics, and lewis split as article
    return articles

# load data
data = ''
for i in range(22):
    name = 'reut2-0' + str(i).zfill(2) + '.sgm'
    data += load_file('reuters21578/' + name)  

stopword_file = 'stopwords.txt' # stopwords file

stopwords = load_stopwords(stopword_file) # load stopwords

# get the articles and their topics
articles = extract_text(data, stopwords) # get articles and their topics