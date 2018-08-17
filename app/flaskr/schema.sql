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
    FOREIGN KEY(userid) REFERENCES user (id)
);


CREATE TABLE recipes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    userid INTEGER NOT NULL,
    url TEXT NOT NULL,
    title VARCHAR(255) NOT NULL,
    FOREIGN KEY(userid) REFERENCES user (id)
);

INSERT INTO users
    (username, password, name)
VALUES
    ('jshuai', 'pbkdf2:sha256:50000$LZplPgoz$7721971950ec7085152a31b8ce22ff0b27c33fae11e1fdfd498d3f64391ea6b3', 'Jonathan');

INSERT INTO recipes
    (userid, url, title) 
VALUES 
    (1, "https://thewoksoflife.com/2016/07/grilled-tandoori-chicken/", "Tandoori Chicken");

INSERT INTO recipes
    (userid, url, title) 
VALUES 
    (1, "https://asianinspirations.com.au/recipe/braised-pork-belly-mao-style", "Pork Belly");

INSERT INTO recipes
    (userid, url, title) 
VALUES 
    (1, "https://www.youtube.com/watch?v=in-c8KE0d8k", "Chicken Cordon Bleu");

INSERT INTO recipes
    (userid, url, title) 
VALUES 
    (1, "https://www.bonappetit.com/recipe/chimichurri-sauce-2", "Chimichurri Sauce");

INSERT INTO restrictions
    (userid, milk, eggs, nuts, peanuts, shellfish, wheat, soy, fish, meat, pork, beef)
VALUES
    (1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0); 


    