from flask import Flask, render_template, request
import openai
import os

app = Flask(__name__)

# Load OpenAI key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    name = request.form['name']
    business = request.form['business']
    goal = request.form['goal']
    grant_type = request.form['grant_type']

    prompt = f"""
    Write a grant proposal for {name}, who owns a business called "{business}". 
    The goal of the grant is: {goal}. 
    The grant type is: {grant_type}.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('result.html', output=result)
