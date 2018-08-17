DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS recipes;
DROP TABLE IF EXISTS restrictions;

CREATE TABLE user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    name TEXT DEFAULT "hungry person"
);

CREATE TABLE restrictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    userid INTEGER NOT NULL,
    milk BIT NOT NULL,
    eggs BIT NOT NULL,
    nuts BIT NOT NULL,
    peanuts BIT NOT NULL,
    shellfish BIT NOT NULL,
    wheat BIT NOT NULL,
    soy BIT NOT NULL,
    fish BIT NOT NULL,
    meat BIT NOT NULL,
    pork BIT NOT NULL,
    beef BIT NOT NULL,
    vegan BIT NOT NULL,
    vegetarian BIT NOT NULL,
    FOREIGN KEY(userid) REFERENCES user (id)
);


CREATE TABLE recipes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    userid INTEGER NOT NULL,
    url TEXT NOT NULL,
    title VARCHAR(255) NOT NULL,
    FOREIGN KEY(userid) REFERENCES user (id)
);
