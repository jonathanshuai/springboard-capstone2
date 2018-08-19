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

CREATE TRIGGER add_restriction
    AFTER INSERT ON users FOR EACH ROW
    BEGIN
        INSERT INTO restrictions
            (userid)
        VALUES
            (NEW.id);
    END;


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