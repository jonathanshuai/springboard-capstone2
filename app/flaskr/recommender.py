import os
from flask import (
    Flask, Blueprint, flash, g, redirect, render_template, request, url_for, send_from_directory
)
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename

from flaskr.auth import login_required
from flaskr.db import get_db

from . import quickrecipe

bp = Blueprint('recommender', __name__)

UPLOAD_FOLDER = 'temp/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


# Function to check if filetype is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# allow image upload and run our application script w/ the image
@bp.route('/', methods=('GET', 'POST'))
@login_required
def index():
    # Get reference to database
    db = get_db()

    # When user uploads a picture...
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file was chosen!')
            return redirect(request.url)
        file = request.files['file']
        
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not file or not allowed_file(file.filename):
            flash('This file type is not allowed.')
            return redirect(request.url)
        else:
            # Upload file to our temp directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename) 
            file.save(file_path)
            imgsrc = url_for('recommender.uploaded_file', filename=filename)

            # Get restrictions for the user
            restrictions = db.execute(
                """
                SELECT 
                    *
                FROM 
                    restrictions
                WHERE
                    userid = ?
                """, (str(g.user['id']))
            ).fetchone()
            
            # Return the restrictions w/ their names and values as tuples in a list
            option_names = restrictions.keys()
            option_values = list(restrictions)
            options = list(zip(option_names, option_values))[2:] # Remove id and userid

            # Blocking call to neural network and recipe api 
            ingredients, recipes = quickrecipe.find_recipes(
                os.path.join(os.getcwd(), UPLOAD_FOLDER, filename), options)

            ingredients = pred_str = ', '.join(list(ingredients))
            if len(recipes) == 0:
                recipes = [{'title': 'No recipes were found', 'url': '', 'image': ''}]
            
            return render_template('recommender/recommender.html', 
                imgsrc=imgsrc, recipes=recipes, ingredients=ingredients)

    # db = get_db()
    return render_template('recommender/recommender.html')


@bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(os.getcwd(), UPLOAD_FOLDER),
                               filename)

@bp.route('/recipes')
@login_required
def recipes():
    # Get the recipes that match the userid
    db = get_db()
    recipe_list = db.execute(
        """
        SELECT 
            *
        FROM 
            recipes
        WHERE
            userid = ?
        """, (str(g.user['id']))
    ).fetchall()

    # Render the recipes page, passing the recipe_list in
    return render_template('recommender/recipes.html', recipe_list=recipe_list)

@bp.route('/settings', methods=('GET', 'POST'))
@login_required
def settings():
    # Get database
    db = get_db()
    
    # If request is a POST, update the database
    if request.method == 'POST':
        # Get request values for each item
        items = ['vegan', 'vegetarian', 'peanut free']
        values = []

        for item in items:
            # check if they were included in the response (we used checkboxes in the HTML)
            if item in request.form:
                values.append(1)
            else:
                values.append(0)
        
        print(list(zip(items, values)))

        db.execute(
            """
            UPDATE 
                restrictions 
            SET 
                vegan=?,
                vegetarian=?,
                peanut_free=?
            WHERE
                userid = ?
            """, (*values, str(g.user['id']))
        )

        db.commit() # Commit the SQL query

    # Get ther restrictions for the user
    restrictions = db.execute(
        """
        SELECT 
            *
        FROM 
            restrictions
        WHERE
            userid = ?
        """, (str(g.user['id']))
    ).fetchone()
    
    # Return the restrictions w/ their names and values as tuples in a list
    option_names = [option_name.replace('_', ' ') for option_name in restrictions.keys()]
    option_values = map(lambda x: "checked" if x == 1 else "", tuple(restrictions))
    options = list(zip(option_names, option_values))[2:] # Remove id and userid

    return render_template('recommender/settings.html', options=options)

@bp.route('/save', methods=['POST'])
def save():
    return "good"