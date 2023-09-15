from pydantic import BaseModel


class GenTopics(BaseModel):
    optimize: bool
    topic_quantity: int


class Topic(BaseModel):
    id: int
    top_words: list[str]
    top_documents: list[list[str]]


class Topics(BaseModel):
    topics: list[Topic]