import os
from flask import Flask, render_template
# from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# supabase: Client = create_client(
#     os.environ.get("SUPABASE_URL"),
#     os.environ.get("SUPABASE_PUBLISHABLE_KEY")
# )

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)