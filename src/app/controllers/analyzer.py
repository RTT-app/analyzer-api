from flask import request, jsonify
from app import app, spec
from flask_pydantic_spec import (Response, Request)

from app.services import (lda_pipeline, translate_topics)
from app.schemas import (GenTopics, Topics)
import pprint
"""
Surgiu a ideia de criar um monitorador de assuntos gerais do /politcs. 
Talvez tudo seja feito com LDA (((((((sei ainda não...)))))))
Seria uma boa ideia monitorar os sentimentos de usuarios com analise de sentimentos (((((mas isso fica no freezer))))).
"""

# - TODO: make schema to return topics

# - TODO: make schema to get topics

@app.post('/get-topics')
@spec.validate(body=Request(GenTopics),resp=Response(HTTP_200=Topics),tags=["LDA Model"])
def gen_topics():
    """
    - Make LDA pipeline to gen the text topics.
    """
    topics = lda_pipeline(request.json)
    topics_ = translate_topics(topics)
    topics_dict = topics_.dict()
    
    return jsonify(topics_dict), 20