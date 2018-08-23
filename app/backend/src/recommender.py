import os

from flask import Flask, Blueprint, jsonify, request, url_for, send_from_directory
from werkzeug.utils import secure_filename

import jwt

from .entities.entity import Session, engine, Base
from .entities.user import User, UserSchema
from .entities.recipe import Recipe, RecipeSchema
from .entities.restriction import Restriction, RestrictionSchema
from . import auth

from . import quickrecipe


bp = Blueprint('recommender', __name__)

UPLOAD_FOLDER = 'temp/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Function to check if filetype is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/recommender', methods=['POST'])
def recommender():
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
            restrictions = session.query(Restriction)\
                .filter(Restriction.userid == userid).one()

            # get the image file from request
            file = request.files['image']

            # ensure correct filetype
            if not file or not allowed_file(file.filename):
                return jsonify({'message': 'File type not allowed.'}), 500
            else:
                # Upload file to our temp directory
                filename = secure_filename(file.filename)
                file_path = os.path.join(os.getcwd(), UPLOAD_FOLDER, filename) 
                file.save(file_path)
                imgsrc = url_for('recommender.uploaded_file', filename=filename)

                # Return the restrictions w/ their names and values as tuples in a list
                options = [
                    ('vegan', restrictions.vegan),
                    ('vegetarian', restrictions.vegetarian),
                    ('peanut_free', restrictions.peanut_free)
                ] 

                ingredients = ''
                # Blocking call to neural network and recipe api 
                ingredients, recipes = quickrecipe.find_recipes(
                    os.path.join(os.getcwd(), UPLOAD_FOLDER, filename), options)

                ingredients = (', '.join(list(ingredients))).replace('_', ' ')

                if len(recipes) == 0:
                    ingredients = 'API overloaded. Please try again later!'

                return jsonify({'ingredients': ingredients, 'recipes': recipes}), 201

    # db = get_db()
    return jsonify({'message': 'end!!'}), 500


@bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(os.getcwd(), UPLOAD_FOLDER),
                               filename)

