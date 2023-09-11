from flask import jsonify
from app import app, spec
from flask_pydantic_spec import (Response, Request)

from app.services.analyzer import lda_pipeline

"""
Surgiu a ideia de criar um monitorador de assuntos gerais do /politcs. 
Talvez tudo seja feito com LDA (((((((sei ainda não...)))))))
Seria uma boa ideia monitorar os sentimentos de usuarios com analise de sentimentos (((((mas isso fica no freezer))))).
"""

# - TODO: make schema to return topics

# - TODO: make schema to get topics

@app.get('/get-topics')
@spec.validate(resp=Response(), tags=["LDA Model"])
def get_topics():
    """
    - Make LDA pipeline to get the topics.
    """
    topics = lda_pipeline()
    return jsonify(), 200