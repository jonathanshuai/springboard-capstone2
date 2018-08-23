#!/bin/bash
export FLASK_APP=./src/main.py
export SECRET_KEY=b'\x10\x06\xc1\x80\xb3\x12g\x1a\xfawK\xc3\x11|\x14\xb5l\xe9g;\xe6&\\;'
source activate fiapp
flask run -h 0.0.0.0
