from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/')
def home():
    return 'Welcome to GrantGenius – Your AI Grant Helper!'

@app.route('/generate', methods=['POST'])
def generate_grant_response():
    # Check if API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({"error": "OpenAI API key not configured"}), 500
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
        
    user_prompt = data.get("prompt")
    if not user_prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    if not user_prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional grant writing assistant. Help the user write winning grant answers with strong logic, detail, and compelling narrative. Focus on clear problem statements, evidence-based solutions, and measurable outcomes."
                },
                {
                    "role": "user", 
                    "content": user_prompt
                }
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        reply = response.choices[0].message.content
        return jsonify({
            "response": reply,
            "model_used": "gpt-4",
            "tokens_used": response.usage.total_tokens if response.usage else None
        })
        
    except Exception as e:
        # Log the error for debugging while returning a user-friendly message
        print(f"Error generating response: {str(e)}")
        return jsonify({"error": "Failed to generate response. Please try again."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment monitoring"""
    return jsonify({"status": "healthy", "service": "GrantGenius"}), 200

if __name__ == '__main__':
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
