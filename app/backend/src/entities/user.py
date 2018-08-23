import datetime
from flask import current_app

from sqlalchemy import Column, String
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

import jwt

from .entity import Entity, Base
from marshmallow import Schema, fields

class User(UserMixin, Entity, Base):
    __tablename__ = 'users'

    username = Column(String(255))
    password = Column(String(255))
    name = Column(String(255))

    def __init__(self, username, password, name, created_by):
        Entity.__init__(self, created_by)
        self.username = username
        self.password = generate_password_hash(password)
        self.name = name

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def encode_auth_token(self):
        payload = {
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1),
            'iat': datetime.datetime.utcnow(),
            'sub': self.id
        }
        return jwt.encode(
            payload,
            current_app.config.get('SECRET_KEY'),
            algorithm='HS256'
        )

    def decode_auth_token(auth_token):
        """Decodes the auth token, returning (userid, message). 
        userid is None if failed.
        
        auth_token    (jwt.something): Auth token to be decoded.
        """

        try:
            payload = jwt.decode(auth_token, current_app.config.get('SECRET_KEY'))
            return payload['sub'], 'Token valid.'
        except jwt.ExpiredSignatureError:
            return None, 'Signature expired. Please log in again.'
        except jwt.InvalidTokenError:
            return None, 'Invalid token. Please log in again.'

class UserSchema(Schema):
    id = fields.Number()
    username = fields.Str()
    password = fields.Str()
    name = fields.Str()
    created_at = fields.DateTime()
    updated_at = fields.DateTime()
    last_updated_by = fields.Str()