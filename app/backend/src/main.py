import os

from flask import Flask, jsonify, request
from flask_cors import CORS

import jwt

from .entities.entity import Session, engine, Base
from .entities.user import User, UserSchema
from .entities.recipe import Recipe, RecipeSchema
from .entities.restriction import Restriction, RestrictionSchema

from . import auth
from . import recommender


# creating the Flask application
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')


CORS(app)

app.register_blueprint(auth.bp)
app.register_blueprint(recommender.bp)

# if needed, generate database schema
Base.metadata.create_all(engine)

@app.route('/recipes')
def get_recipes():
    # Check authentication and get userid
    auth_header = request.headers.get('Authorization')
    
    auth_token = ''
    if auth_header:
        auth_token = auth_header.split(" ")[1]
    
    if auth_token:
        userid, message = User.decode_auth_token(auth_token)
        # On failure, return error message
        if not userid:
            return jsonify({'message': message}), 401
        # On successful auth
        else:
            # fetching from the database relevant recipes
            session = Session()
            recipe_objects = session.query(Recipe)\
                .filter(Recipe.userid == userid).all()

            # Turn it into some json object and return
            schema = RecipeSchema(many=True)
            recipes = schema.dump(recipe_objects)

            session.close()
            return jsonify(recipes.data), 201

@app.route('/save_recipe', methods=['POST'])
def save_recipe():
    # Check authentication and get userid
    auth_header = request.headers.get('Authorization')
    
    auth_token = ''
    if auth_header:
        auth_token = auth_header.split(" ")[1]
    
    if auth_token:
        userid, message = User.decode_auth_token(auth_token)
        # On failure, return error message
        if not userid:
            return jsonify({'message': message}), 401
        # On successful auth
        else:
            # Create recipe 
            session = Session()
            posted_recipe = RecipeSchema(only=('title', 'url', 'imgsrc'))\
                .load(request.get_json())

            print(posted_recipe.data)
            print(request.get_json())
            recipe = Recipe(**posted_recipe.data, userid=userid, created_by="HTTP post request")

            # persist recipe
            session.add(recipe)
            session.commit()

            # close session and return good
            session.close()

            return jsonify({'message': 'good'}), 201

@app.route('/delete_recipe', methods=['POST'])
def delete_recipe():
    # Check authentication and get userid
    auth_header = request.headers.get('Authorization')
    
    auth_token = ''
    if auth_header:
        auth_token = auth_header.split(" ")[1]
    
    if auth_token:
        userid, message = User.decode_auth_token(auth_token)
        # On failure, return error message
        if not userid:
            return jsonify({'message': message}), 401
        # On successful auth
        else:
            # Create recipe 
            session = Session()
            request_json = request.get_json()
            session.query(Recipe).filter(Recipe.userid == userid, 
                    Recipe.title == request_json['title']).delete()

            # persist recipe
            session.commit()

            # close session and return good
            session.close()

            return jsonify({'message': 'good'}), 201

@app.route('/restrictions')
def get_restrictions():
    # Check authentication and get userid
    auth_header = request.headers.get('Authorization')
    
    auth_token = ''
    if auth_header:
        auth_token = auth_header.split(" ")[1]
    
    if auth_token:
        userid, message = User.decode_auth_token(auth_token)
        # On failure, return error message
        if not userid:
            return jsonify({'message': message}), 401
        # On successful auth
        else:
            # fetching from the database restrictions
            session = Session()
            restrictions_object = session.query(Restriction)\
                .filter(Restriction.userid == userid).one()

            # Turn it into some json object and return
            schema = RestrictionSchema()
            restrictions = schema.dump(restrictions_object)

            session.close()
            return jsonify(restrictions.data), 201

@app.route('/update_restriction', methods=['POST'])
def update_restriction():
    # Check authentication and get userid
    auth_header = request.headers.get('Authorization')
    
    auth_token = ''
    if auth_header:
        auth_token = auth_header.split(" ")[1]
    
    if auth_token:
        userid, message = User.decode_auth_token(auth_token)
        # On failure, return error message
        if not userid:
            return jsonify({'message': message}), 401
        # On successful auth
        else:
            # fetching from the database relevant recipes
            session = Session()
            restriction_object = session.query(Restriction)\
                .filter(Restriction.userid == userid).one()

            request_json = request.get_json()

            restriction_object.vegan = request_json['vegan']
            restriction_object.vegetarian = request_json['vegetarian']
            restriction_object.peanut_free = request_json['peanut_free']

            session.commit()
            session.close()

            return jsonify({'message': 'good'}), 201

@app.route('/dummy', methods=['POST'])
def dummy():
    auth_header = request.headers.get('Authorization')
    
    auth_token = ''
    if auth_header:
        auth_token = auth_header.split(" ")[1]
    
    if auth_token:
        userid, message = User.decode_auth_token(auth_token)
        if userid:
            print(userid) # This works!
            return jsonify({'message': message}), 201
        else:
            return jsonify({'message': message}), 401
