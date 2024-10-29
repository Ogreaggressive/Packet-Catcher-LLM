from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from backendModel import initialize, run_query

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Initialize the model and QA system
initialize()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    if request.is_json:
        user_query = request.json.get('query')
    else:
        user_query = request.form.get('query')

    if not user_query:
        return jsonify({'error': 'Missing query parameter'}), 400

    response = run_query(user_query)
    response_string = str(response)

    return jsonify({"response": response_string})

if __name__ == '__main__':
    app.run(debug=True)

