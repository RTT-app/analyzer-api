# -*- coding: utf-8 -*-
"""Tópicos com LDA - Reddit Analyser

# Modelagem de Tópicos com LDA

Estudos LDA para dados do Reddit.
"""

#from google.colab import drive
#drive.mount('/content/drive')

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# clean comments function
def clean_comment(comment):
  url_pattern = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
  comment = comment.lower()
  comment = re.sub(url_pattern," ", comment)
  comment = re.sub("(\\d|\\W)+|\w*\d\w*"," ", comment)
  comment = re.sub('\(?\dx\)?',' ',comment)
  comment = re.sub('\d+',' ',comment)
  comment = re.sub('_',' ',comment)
  
  return comment

# LDA functions
def print_top_words(model, feature_names, n_top_words):
  for topic_idx, topic in enumerate(model.components_):
    print("\n--\nTopic #{}: ".format(topic_idx + 1))
    message = ", ".join([feature_names[i]
                          for i in topic.argsort()[:-n_top_words - 1:-1]])
    print(message)
  print()

def display_topics(W, H, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("\n--\nTopic #{}: ".format(topic_idx + 1))
        print(", ".join([feature_names[i]
                for i in topic.argsort()[:-no_top_words - 1:-1]]).upper())
        print()
        top_d_idx = np.argsort(W[:,topic_idx])[::-1][0:no_top_documents]
        for d in top_d_idx: 
          doc_data = documents[['title', 'clean_comment']].iloc[d]
          print('{} - {} : \t{:.2f}'.format(doc_data[1], doc_data[0], W[d, topic_idx]))


nltk.download('stopwords')

df = pd.read_csv('../data/exp.csv')

df.drop(['self_text','Unnamed: 0'], axis = 1, inplace = True)

stop_words = list(stopwords.words('english'))

df['clean_comment'] = df.comment.apply(clean_comment)

df = df[['title','comment','clean_comment','score']]

"""vectorizer"""

tf_vectorizer = CountVectorizer(stop_words = stop_words)
vect = tf_vectorizer.fit_transform(df.clean_comment)

words = tf_vectorizer.get_feature_names_out()


# Grid Search 
search_params = {'n_components': [5, 7, 9, 11, 13, 15]}

# Init the Model
lda = LatentDirichletAllocation(learning_method='online')

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(vect)

best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(vect))

lda = LatentDirichletAllocation(n_components = 5, learning_method='online')
lda.fit(vect)
doc_topic_matrix = lda.transform(vect)

display_topics(doc_topic_matrix,
               lda.components_, 
               words,
               df,
               15, 
               10)