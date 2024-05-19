from flask import Flask, render_template, request, jsonify
from transformers_summarizer import summarize_text

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def get_summary():
    data = request.get_json()
    text = data['text']
    summary = summarize_text(text)
    return jsonify({'summary': summary})

if __name__ == "__main__":
    app.run(debug=True)
