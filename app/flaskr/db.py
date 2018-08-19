import sqlite3

import pymysql

import click
from flask import current_app, g
from flask.cli import with_appcontext

# Copied from flask tutorial, database connection
def get_db():
    if 'db' not in g:
        g.db = pymysql.connect(
            host=current_app.config['DB_HOST'], 
            user=current_app.config['DB_USER'],
            passwd=current_app.config['DB_PASS'],
            db=current_app.config['DB_NAME']
        ).cursor(pymysql.cursors.DictCursor)

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_db():
    db = get_db()

    db.execute("DROP TABLE IF EXISTS users;")
    db.execute("DROP TABLE IF EXISTS recipes;")
    db.execute("DROP TABLE IF EXISTS restrictions;")

    db.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            username VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(500) NOT NULL,
            name VARCHAR(255)
        );
    """)
    db.execute("""
    CREATE TABLE restrictions (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        userid INTEGER NOT NULL,
        vegan BIT DEFAULT 0,
        vegetarian BIT DEFAULT 0,
        peanut_free BIT DEFAULT 0
    );
    """)
    db.execute("""
    CREATE TABLE recipes (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        userid INTEGER NOT NULL,
        url VARCHAR(500) NOT NULL,
        title VARCHAR(255) NOT NULL
    );
    """)
    db.execute("""
    CREATE TRIGGER add_restriction
        AFTER INSERT ON users FOR EACH ROW
        BEGIN
            INSERT INTO restrictions
                (userid)
            VALUES
                    (NEW.id);
    END;
    """)

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
