from flask import Flask, request, jsonify, render_template, redirect, url_for
from openai import OpenAI
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/')
def home():
    """Main page with form to submit grant requests"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page explaining GrantGenius"""
    return render_template('about.html')

@app.route('/result')
def result():
    """Display result page (typically reached after generation)"""
    # This could be used for direct access to result page
    return render_template('result.html', output="No content generated yet. Please submit a request first.")

@app.route('/generate', methods=['POST'])
def generate_grant_response():
    # Check if API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OpenAI API key not configured")
        return jsonify({"error": "OpenAI API key not configured"}), 500
    
    # Handle both JSON API calls and form submissions
    if request.is_json:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        user_prompt = data.get("prompt")
        return_json = True
    else:
        # Handle form submission
        user_prompt = request.form.get("prompt")
        return_json = False
    
    if not user_prompt:
        error_msg = "Prompt is required"
        if return_json:
            return jsonify({"error": error_msg}), 400
        else:
            return render_template('index.html', error=error_msg)
    
    if not user_prompt.strip():
        error_msg = "Prompt cannot be empty"
        if return_json:
            return jsonify({"error": error_msg}), 400
        else:
            return render_template('index.html', error=error_msg)
    
    # Optional parameters (only for JSON requests)
    max_tokens = 1500
    temperature = 0.7
    if return_json and request.is_json:
        data = request.get_json()
        max_tokens = data.get("max_tokens", 1500)
        temperature = data.get("temperature", 0.7)
    
    try:
        logger.info(f"Generating response for prompt length: {len(user_prompt)}")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a professional grant writing assistant with expertise in creating compelling, well-structured grant proposals. Help users write winning grant applications by:

1. Providing clear, compelling problem statements
2. Developing evidence-based solutions with specific methodologies
3. Creating measurable outcomes and evaluation metrics
4. Using persuasive language that demonstrates impact
5. Structuring responses with logical flow and strong narrative
6. Including relevant data, statistics, and research when appropriate
7. Addressing potential concerns or limitations proactively

Always focus on the specific requirements of the grant opportunity and tailor your response accordingly."""
                },
                {
                    "role": "user", 
                    "content": user_prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        reply = response.choices[0].message.content
        
        # Log successful generation
        logger.info(f"Successfully generated response with {response.usage.total_tokens if response.usage else 'unknown'} tokens")
        
        if return_json:
            return jsonify({
                "response": reply,
                "model_used": "gpt-4",
                "tokens_used": response.usage.total_tokens if response.usage else None,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None
            })
        else:
            return render_template('result.html', output=reply, prompt=user_prompt)
        
    except Exception as e:
        # Log the error for debugging
        logger.error(f"Error generating response: {str(e)}")
        
        # Return appropriate error response
        if "rate_limit" in str(e).lower():
            error_msg = "Rate limit exceeded. Please try again in a moment."
            status_code = 429
        elif "insufficient_quota" in str(e).lower():
            error_msg = "API quota exceeded. Please check your OpenAI account."
            status_code = 402
        else:
            error_msg = "Failed to generate response. Please try again."
            status_code = 500
        
        if return_json:
            return jsonify({"error": error_msg}), status_code
        else:
            return render_template('index.html', error=error_msg)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment monitoring"""
    try:
        # Basic health check
        api_key_configured = bool(os.getenv("OPENAI_API_KEY"))
        
        return jsonify({
            "status": "healthy",
            "service": "GrantGenius",
            "api_key_configured": api_key_configured,
            "timestamp": None  # Could add timestamp if needed
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def api_generate_grant_response():
    """API-only endpoint for JSON requests"""
    # Check if API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OpenAI API key not configured")
        return jsonify({"error": "OpenAI API key not configured"}), 500
    
    # Validate request data
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
        
    user_prompt = data.get("prompt")
    if not user_prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    if not user_prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400
    
    # Optional parameters
    max_tokens = data.get("max_tokens", 1500)
    temperature = data.get("temperature", 0.7)
    
    try:
        logger.info(f"API: Generating response for prompt length: {len(user_prompt)}")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a professional grant writing assistant with expertise in creating compelling, well-structured grant proposals. Help users write winning grant applications by:

1. Providing clear, compelling problem statements
2. Developing evidence-based solutions with specific methodologies
3. Creating measurable outcomes and evaluation metrics
4. Using persuasive language that demonstrates impact
5. Structuring responses with logical flow and strong narrative
6. Including relevant data, statistics, and research when appropriate
7. Addressing potential concerns or limitations proactively

Always focus on the specific requirements of the grant opportunity and tailor your response accordingly."""
                },
                {
                    "role": "user", 
                    "content": user_prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        reply = response.choices[0].message.content
        
        # Log successful generation
        logger.info(f"API: Successfully generated response with {response.usage.total_tokens if response.usage else 'unknown'} tokens")
        
        return jsonify({
            "response": reply,
            "model_used": "gpt-4",
            "tokens_used": response.usage.total_tokens if response.usage else None,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None
        })
        
    except Exception as e:
        # Log the error for debugging
        logger.error(f"API: Error generating response: {str(e)}")
        
        # Return user-friendly error message
        if "rate_limit" in str(e).lower():
            return jsonify({"error": "Rate limit exceeded. Please try again in a moment."}), 429
        elif "insufficient_quota" in str(e).lower():
            return jsonify({"error": "API quota exceeded. Please check your OpenAI account."}), 402
        else:
            return jsonify({"error": "Failed to generate response. Please try again."}), 500

@app.route('/templates/page')
def templates_page():
    """Display templates page with examples"""
    templates = {
        "problem_statement": {
            "title": "Problem Statement",
            "description": "Create a compelling problem statement that clearly articulates the need",
            "template": "Help me write a compelling problem statement for a grant proposal about [TOPIC]. Include relevant statistics, evidence of need, and clear articulation of the gap this grant will address."
        },
        "methodology": {
            "title": "Methodology",
            "description": "Develop a detailed methodology section with specific steps and timeline",
            "template": "Help me develop a detailed methodology section for a grant proposal that will [OBJECTIVE]. Include specific steps, timeline, and evaluation methods."
        },
        "budget_justification": {
            "title": "Budget Justification", 
            "description": "Write a comprehensive budget justification explaining costs",
            "template": "Help me write a budget justification for [PROJECT TYPE] that costs $[AMOUNT]. Explain why each expense is necessary and how it contributes to project success."
        },
        "evaluation_plan": {
            "title": "Evaluation Plan",
            "description": "Create both formative and summative evaluation methods",
            "template": "Help me create an evaluation plan for a grant project that aims to [GOAL]. Include both formative and summative evaluation methods with specific metrics."
        },
        "sustainability": {
            "title": "Sustainability Plan",
            "description": "Explain how the project will continue beyond the grant period",
            "template": "Help me write a sustainability plan explaining how the project funded by this grant will continue beyond the grant period."
        }
    }
    return render_template('templates.html', templates=templates)

@app.route('/api/templates', methods=['GET'])
def get_templates():
    """API endpoint - Provide common grant writing templates/prompts"""
    templates = {
        "problem_statement": "Help me write a compelling problem statement for a grant proposal about [TOPIC]. Include relevant statistics, evidence of need, and clear articulation of the gap this grant will address.",
        "methodology": "Help me develop a detailed methodology section for a grant proposal that will [OBJECTIVE]. Include specific steps, timeline, and evaluation methods.",
        "budget_justification": "Help me write a budget justification for [PROJECT TYPE] that costs $[AMOUNT]. Explain why each expense is necessary and how it contributes to project success.",
        "evaluation_plan": "Help me create an evaluation plan for a grant project that aims to [GOAL]. Include both formative and summative evaluation methods with specific metrics.",
        "sustainability": "Help me write a sustainability plan explaining how the project funded by this grant will continue beyond the grant period."
    }
    
    return jsonify({"templates": templates})

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set")
    
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting GrantGenius on port {port} (debug: {debug_mode})")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
