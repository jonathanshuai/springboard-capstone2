import functools

from flask import Blueprint, Flask, jsonify, request

from werkzeug.security import generate_password_hash

from .entities.entity import Session, engine, Base
from .entities.user import User, UserSchema
from .entities.restriction import Restriction

bp = Blueprint('auth', __name__)

@bp.route('/register', methods=['POST'])
def register():
    # mount user object
    posted_user = UserSchema(only=('username', 'password', 'name'))\
        .load(request.get_json())

    session = Session()
    same_username = session.query(User)\
        .filter(User.username == posted_user.data['username']).all()

    if len(same_username) != 0:
        session.close()
        return 'That username already exists.', 500

    user = User(**posted_user.data, created_by="HTTP post request")

    # persist user
    session.add(user)
    session.commit()

    # return created user
    new_user = UserSchema().dump(user).data


    # Create restriction for this user
    restriction = Restriction(new_user['id'], created_by="HTTP post request")

    session.add(restriction)
    session.commit()
    session.close()
    return jsonify(new_user), 201

@bp.route('/authenticate', methods=['POST'])
def login():
    # Get JSON from request: it should have username and password
    request_json = request.get_json()

    session = Session()
    user = session.query(User)\
        .filter(User.username == request_json['username']).one_or_none()

    if user is None:
        return 'No user with that username.', 500
    if not user.check_password(request_json['password']):
        return 'Password is incorrect.', 500

    # credentials have been validated; 
    # Return jwt auth token
    auth_token = user.encode_auth_token()
    responseObject = {
        'status': 'success',
        'message': 'Successfully logged in.',
        'auth_token': auth_token.decode()
    }

    return jsonify(responseObject), 200 