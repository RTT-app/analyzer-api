import requests
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from config import ( 
                    DB_API_HOST, 
                    DB_API_PORT
                    )


nltk.download('stopwords')

# - AUX functions.
def get_top_words_p_topic(words, topics, feature_names, documents, no_top_words, no_top_documents):
    # - TODO: solve get_top_words_p_topic return
    
    for topic_idx, topic in enumerate(topics):
        print(f"\n--\nTopic #{topic_idx + 1}: ")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]).upper())
        print()
        top_d_idx = np.argsort(words[:,topic_idx])[::-1][0:no_top_documents]
        for d in top_d_idx:
          doc_data = documents[['title', 'clean_comment']].iloc[d]
          print('{} - {} : \t{:.2f}'.format(doc_data[1], doc_data[0], words[d, topic_idx]))


# - PIPELINE functions.
def data_igestion():
  db_url = f'http://{DB_API_HOST}:{DB_API_PORT}/get-posts'
  
  response = requests.get(db_url)

  content = response.json()['posts']
  
  return content


def json_to_df(json_data:str):
    # - TODO.
    pass


def vectorize_data(df):
  #vectorizer
  stop_words = list(stopwords.words('english'))
  tf_vectorizer = CountVectorizer(stop_words=stop_words)
  vect = tf_vectorizer.fit_transform(df.clean_comment)

  words = tf_vectorizer.get_feature_names_out()

  return words, vect


def get_best_params(vect, n_components=[3, 5, 7, 9]):
  # Define Search Param
  search_params = {'n_components': n_components}
  
  lda = LatentDirichletAllocation(learning_method='online')

  model = GridSearchCV(lda, param_grid=search_params)
  model.fit(vect)

  params = {
                "best_model": model.best_estimator_,
                "best_model_params": model.best_params_
                }
  
  return params


def get_topics(words, vect, data_frame:pd.core.frame.DataFrame,
               params={"best_model": None,
                       "best_model_params": None}
                       ) -> dict:
  
  # - TODO: solve get_top_words_p_topic return

  n_components = params.get("best_model_params").get("n_components") if params.get("best_model_params") else 5
  
  lda = params.get("best_model") if params.get("best_model_params") else None
  
  if not lda:
    lda = LatentDirichletAllocation(learning_method='online')
    lda = LatentDirichletAllocation(n_components=n_components, learning_method='online')

  lda.fit(vect)
  doc_topic_matrix = lda.transform(vect)

  topics = get_top_words_p_topic(doc_topic_matrix,
               lda.components_,
               words,
               data_frame,
               15,
               10)
  
  topics = {}

  return topics


def lda_pipeline(request_info:dict):
  # - TODO: define pipeline structure.
  pass