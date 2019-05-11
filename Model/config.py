from flask import Flask
from flask_mongoalchemy import BaseQuery
from flask_mongoalchemy import MongoAlchemy
app = Flask(__name__)
app.config['MONGOALCHEMY_DATABASE'] = 'dicecoin'
db = MongoAlchemy(app)

class Seed(db.Document):
    Sseed = db.StringField()
    Cseed = db.StringField()
class DiceData(db.Document):
    query_class= BaseQuery
    number = db.FloatField()
    over   =db.BoolField() 
    time   =db.DateTimeField()
    target = db.FloatField()
   
   
