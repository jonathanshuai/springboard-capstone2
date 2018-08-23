from sqlalchemy import Column, Integer, String, ForeignKey

from .entity import Entity, Base
from marshmallow import Schema, fields

class Recipe(Entity, Base):
    __tablename__ = 'recipes'

    userid = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255))
    url = Column(String(1023))
    imgsrc = Column(String(1023))

    def __init__(self, userid, title, url, imgsrc, created_by):
        Entity.__init__(self, created_by)
        self.userid = userid
        self.title = title
        self.url = url
        self.imgsrc = imgsrc

        from sqlalchemy import Column, String

class RecipeSchema(Schema):
    id = fields.Number()
    userid = fields.Number()
    title = fields.Str()
    url = fields.Str()
    imgsrc = fields.Str()
    created_at = fields.DateTime()
    updated_at = fields.DateTime()
    last_updated_by = fields.Str()