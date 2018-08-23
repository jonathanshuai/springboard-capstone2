from sqlalchemy import Column, Integer, Boolean, ForeignKey

from .entity import Entity, Base
from marshmallow import Schema, fields

class Restriction(Entity, Base):
    __tablename__ = 'restrictions'

    userid = Column(Integer, ForeignKey("users.id"), nullable=False)
    vegan = Column(Boolean)
    vegetarian = Column(Boolean)
    peanut_free = Column(Boolean)

    def __init__(self, userid, created_by, 
        vegan=False, vegetarian=False, peanut_free=False):
        
        Entity.__init__(self, created_by)
        self.userid = userid
        self.vegan = vegan
        self.vegetarian = vegetarian
        self.peanut_free = peanut_free

class RestrictionSchema(Schema):
    id = fields.Number()
    userid = fields.Number()
    vegan = fields.Boolean()
    vegetarian = fields.Boolean()
    peanut_free = fields.Boolean()
    created_at = fields.DateTime()
    updated_at = fields.DateTime()
    last_updated_by = fields.Str()