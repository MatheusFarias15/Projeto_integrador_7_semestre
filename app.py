import os
from flask import Flask, render_template
from dotenv import load_dotenv
from routes.glucose import glucose_routes 

load_dotenv()

app = Flask(__name__)

# Registrando as rotas no app principal
app.register_blueprint(glucose_routes)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)