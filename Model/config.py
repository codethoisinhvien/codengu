from flask import Flask
from flask_mongoalchemy import BaseQuery
from flask_mongoalchemy import MongoAlchemy
app = Flask(__name__)
app.config['MONGOALCHEMY_DATABASE'] = 'heroku_cjxxw05p'
app.config['MONGOALCHEMY_CONNECTION_STRING']="mongodb://root:Phongthien1308@ds247637.mlab.com:47637/heroku_cjxxw05p"
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
   
   
