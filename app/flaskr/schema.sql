DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS recipes;
DROP TABLE IF EXISTS restrictions;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    name TEXT DEFAULT "hungry person"
);

CREATE TABLE restrictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    userid INTEGER NOT NULL,
    vegan BIT DEFAULT 0,
    vegetarian BIT DEFAULT 0,
    peanut_free BIT DEFAULT 0,
    FOREIGN KEY(userid) REFERENCES user (id)
);


CREATE TABLE recipes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    userid INTEGER NOT NULL,
    url TEXT NOT NULL,
    -- imgurl TEXT NOT NULL,
    title VARCHAR(255) NOT NULL,
    FOREIGN KEY(userid) REFERENCES user (id)
);