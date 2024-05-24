from flask import Flask, render_template, request, jsonify
from transformers_summarizer import summarize_text
import mysql.connector

app = Flask(__name__)

# Initialize MySQL connection
mydb = mysql.connector.connect(
    host="localhost",
    port="3306",
    user="root",
    password="JeffConson210!",
    database="summaryme"
)

# Create cursor
mycursor = mydb.cursor()

@app.route('/')
def home():
    try:
        # Fetch recent chat history from the database
        mycursor.execute("SELECT * FROM chat_history ORDER BY timestamp DESC LIMIT 10")
        chat_history = mycursor.fetchall()
        return render_template('index.html', chat_history=chat_history)
    except Exception as e:
        print("Error fetching chat history:", e)
        return "Error fetching chat history"

@app.route('/summarize', methods=['POST'])
def get_summary():
    data = request.get_json()
    text = data['text']
    summary = summarize_text(text)
   
    # Insert user message into the database
    sql = "INSERT INTO chat_history (sender, message) VALUES (%s, %s)"
    val = ("user", text)
    mycursor.execute(sql, val)
    mydb.commit()
   
    # Insert bot message into the database
    sql = "INSERT INTO chat_history (sender, message) VALUES (%s, %s)"
    val = ("bot", summary)
    mycursor.execute(sql, val)
    mydb.commit()
   
    return jsonify({'summary': summary})

if __name__ == "__main__":
    app.run(debug=True)
