import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from nltk.probability import FreqDist
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Load the dataset
data = pd.read_csv('processed-abcnews.csv')

# Convert the 'headline_text' column to a list of lists of tokens
headlines = data['headline_text']
word_lists = []
for headline in headlines:
    word_list = ast.literal_eval(headline)
    word_lists.append(word_list)

# Flatten the list of lists into a single list
flat_tokens = [word for sublist in word_lists for word in sublist]

# Calculate the frequency distribution of words
frequency_dist = FreqDist(flat_tokens)

# Bar plot of the top 30 most frequent words
sorted_freq_dist = sorted(frequency_dist.items(), key=lambda k: k[1], reverse=True)
sorted_freq_dist = sorted_freq_dist[:30]
plt.figure(figsize=(10, 5))
plt.bar([i[0] for i in sorted_freq_dist], [i[1] for i in sorted_freq_dist])
plt.xticks(rotation=90)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 30 Most Frequent Words')
plt.show()

# Histogram of headline word counts
headline_word_counts = [len(tokens) for tokens in word_lists]
plt.figure(figsize=(10, 5))
plt.hist(headline_word_counts, bins=20, color='skyblue')
plt.title('Histogram of Headline Word Counts')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

# Word count distribution box plot
plt.figure(figsize=(10, 5))
sns.boxplot(headline_word_counts)
plt.title('Word Count Distribution')
plt.xlabel('Word Count')
plt.show()

years = [str(date)[:4] for date in data['publish_date']]

year_counts = pd.Series(years).value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5))
year_counts.plot(kind='bar', ax=ax)
ax.set_xlabel('Year')
ax.set_ylabel('Count')
ax.set_title('Number of Articles by Year')
plt.show()