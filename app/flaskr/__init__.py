import os

from flask import Flask
from . import db
from . import auth
from . import recommender

from dotenv import load_dotenv
load_dotenv()

 # prints mint

def create_app(test_config=None):
    # Create and configure the Flask app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
        # DB_HOST=os.environ.get('DB_HOST'), 
        # DB_USER=os.environ.get('DB_USER'),
        # DB_PASS=os.environ.get('DB_PASS'),
        # DB_NAME=os.environ.get('DB_NAME')
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    # Initialize database with app
    db.init_app(app)

    # Register blueprint for authentication
    app.register_blueprint(auth.bp)

    # Register the blueprint for recommender application (main app)
    app.register_blueprint(recommender.bp)
    app.add_url_rule('/', endpoint='index')
    
    return app