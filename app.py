from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/')
def home():
    return 'Welcome to GrantGenius – Your AI Grant Helper!'

@app.route('/generate', methods=['POST'])
def generate_grant_response():
    data = request.get_json()
    user_prompt = data.get("prompt")

    if not user_prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # ← Using GPT-4 here
            messages=[
                {"role": "system", "content": "You are a professional grant writing assistant. Help the user write winning grant answers with strong logic and detail."},
                {"role": "user", "content": user_prompt}
            ]
        )
        reply = response.choices[0].message.content
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

