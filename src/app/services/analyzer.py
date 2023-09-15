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

from app.schemas import (Topic, Topics)

nltk.download('stopwords')


#   get_top_words_p_topic(doc_topic_matrix, lda.components_, words, data_frame, 15, 10)
def get_top_words_p_topic(words, topics, feature_names, documents, no_top_words, no_top_documents):
  # - TODO: solve get_top_words_p_topic return
  topics_list = []

  for topic_idx, topic in enumerate(topics):
    topics_list.append({'id':topic_idx + 1,"top_words":[], "top_documents":[]})
    
    top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    topics_list[-1]["top_words"] = top_words
    top_d_idx = np.argsort(words[:,topic_idx])[::-1][0:no_top_documents]
    for d in top_d_idx:
      doc_data = documents[['title', 'comment']].iloc[d]
                                        # tuple doc: post title,  comment,     topic percent
      topics_list[-1]["top_documents"].append([doc_data[1], doc_data[0], str(words[d, topic_idx])])
  
  return topics_list


def format_to_valid_dict(json_data):
  valid_dict = {
                "comment":[],
                "score":[],
                "self_text":[],
                "title":[],
               }

  for post in json_data:
    valid_dict["comment"].append(post["comment"])
    valid_dict["score"].append(post["score"])
    valid_dict["self_text"].append(post["self_text"])
    valid_dict["title"].append(post["title"])
  
  return valid_dict


# - PIPELINE functions.
def data_igestion():
  db_url = f'http://{DB_API_HOST}:{DB_API_PORT}/get-posts'
  
  response = requests.get(db_url)

  content = response.json()['posts']
  
  return content


def json_to_df(json_data):
  valid_dict = format_to_valid_dict(json_data)
  dataframe = pd.DataFrame.from_dict(valid_dict)
  
  return dataframe


def vectorize_data(df):
  #vectorizer
  stop_words = list(stopwords.words('english'))
  tf_vectorizer = CountVectorizer(stop_words=stop_words)
  vect = tf_vectorizer.fit_transform(df.comment)

  words = tf_vectorizer.get_feature_names_out()

  return words, vect


def get_topics(words, vect, n_components, data_frame, params):
  # - TODO: solve get_top_words_p_topic return
  n_components = params.get("best_model_params").get("n_components") if params.get("best_model_params") else n_components
  
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
  
  return topics


# - AUX functions.
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


def translate_topics(topics):
  topics_ = []

  for topic in topics:
    topic_ = Topic.parse_obj(topic)
    topics_.append(topic_)
  
  topics__ = Topics(topics=topics_)
  return topics__


def lda_pipeline(request_info:dict):
  # - TODO: define pipeline structure.
  params = {
    "best_model": None,
    "best_model_params": {
      'n_components':request_info['topic_quantity']
      }
    }

  json_data = data_igestion()
  data_frame = json_to_df(json_data)
  words, vect = vectorize_data(data_frame)
  
  if request_info['optimize']:
    params = get_best_params(vect=vect)

  topics = get_topics(words, vect, params.get("best_model_params").get("topic_quantity"), data_frame, params=params)

  return topics 