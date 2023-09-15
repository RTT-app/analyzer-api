from flask import Flask
from flask_pydantic_spec import FlaskPydanticSpec

app = Flask(__name__)
spec = FlaskPydanticSpec('analyzer-api')
spec.register(app)

from app.controllers import analyzer