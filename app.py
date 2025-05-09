from flask import Flask, request, jsonify
from podcast_llm import PodcastLLM
import os

app = Flask(__name__)
llm = PodcastLLM(os.getenv('OPENAI_API_KEY'))

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing question in request body'}), 400
        
    try:
        response = llm.ask(data['question'])
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 