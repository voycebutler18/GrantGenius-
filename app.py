from flask import Flask, request, jsonify, render_template, redirect, url_for
from openai import OpenAI
from flask import Flask, request, jsonify
import os
import logging
import requests
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Load sensitive credentials from Render environment variables
API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("GOOGLE_CSE_ID")

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
import requests
import json
from flask import request, jsonify
import logging

logger = logging.getLogger(__name__)

@app.route('/search', methods=['GET'])
def search_grants():
    """Search for grants across the entire web - foundations, corporations, nonprofits, government"""
    query = request.args.get('query', '')
    location = request.args.get('location', '')
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    try:
        # Build comprehensive search queries for different types of grants
        search_queries = []
        
        # Base grant search
        base_query = f"{query} grants funding opportunities"
        if location:
            base_query += f" {location}"
        search_queries.append(base_query)
        
        # Foundation grants
        foundation_query = f"{query} foundation grants private funding"
        if location:
            foundation_query += f" {location}"
        search_queries.append(foundation_query)
        
        # Corporate grants
        corporate_query = f"{query} corporate grants business funding sponsorship"
        if location:
            corporate_query += f" {location}"
        search_queries.append(corporate_query)
        
        # Nonprofit grants
        nonprofit_query = f"{query} nonprofit grants charitable funding"
        if location:
            nonprofit_query += f" {location}"
        search_queries.append(nonprofit_query)
        
        all_results = []
        
        # Perform multiple searches to get comprehensive results
        for search_query in search_queries:
            try:
                # Use a web search API (you'll need to configure this with your preferred service)
                # Example using SerpAPI, Bing Search API, or Google Custom Search
                search_results = perform_web_search(search_query)
                
                if search_results and 'items' in search_results:
                    # Filter and format results
                    filtered_results = filter_grant_results(search_results['items'], query)
                    all_results.extend(filtered_results)
                    
            except Exception as search_error:
                logger.warning(f"Search failed for query '{search_query}': {str(search_error)}")
                continue
        
        # Remove duplicates and rank results
        unique_results = remove_duplicates(all_results)
        ranked_results = rank_grant_results(unique_results, query, location)
        
        # Limit to top 20 results
        final_results = ranked_results[:20]
        
        logger.info(f"Found {len(final_results)} grant opportunities for: {query}")
        
        return jsonify({
            "query": query,
            "location": location,
            "total_results": len(final_results),
            "items": final_results
        })
        
    except Exception as e:
        logger.error(f"Grant search error: {str(e)}")
        return jsonify({
            "error": "Search service temporarily unavailable",
            "items": []
        }), 500

def perform_web_search(query):
    """Perform actual web search using your preferred search API"""
    
    # Option 1: Google Custom Search API
    # GOOGLE_API_KEY = "your_google_api_key"
    # SEARCH_ENGINE_ID = "your_search_engine_id"
    # url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}"
    
    # Option 2: Bing Search API
    # BING_API_KEY = "your_bing_api_key"
    # url = "https://api.bing.microsoft.com/v7.0/search"
    # headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    # params = {"q": query, "count": 10}
    
    # Option 3: SerpAPI
    SERPAPI_KEY = "your_serpapi_key"  # Get from serpapi.com
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "num": 10
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Format results based on API response structure
        if "organic_results" in data:  # SerpAPI format
            formatted_results = {
                "items": [
                    {
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", "")
                    }
                    for result in data["organic_results"]
                ]
            }
            return formatted_results
        elif "webPages" in data:  # Bing format
            return {
                "items": [
                    {
                        "title": item.get("name", ""),
                        "link": item.get("url", ""),
                        "snippet": item.get("snippet", "")
                    }
                    for item in data["webPages"]["value"]
                ]
            }
        elif "items" in data:  # Google Custom Search format
            return data
            
    except requests.RequestException as e:
        logger.error(f"Web search API error: {str(e)}")
        raise

def filter_grant_results(results, query):
    """Filter search results to focus on actual grant opportunities"""
    
    # Keywords that indicate legitimate grant opportunities
    grant_keywords = [
        "grant", "funding", "fellowship", "scholarship", "award", "foundation",
        "nonprofit", "charitable", "donation", "sponsor", "financial support",
        "apply", "application", "deadline", "eligibility", "requirements"
    ]
    
    # Keywords that indicate scams or irrelevant results
    exclude_keywords = [
        "loan", "credit", "debt", "personal finance", "get rich quick",
        "guaranteed", "no application", "instant approval", "scam"
    ]
    
    filtered = []
    
    for result in results:
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        combined_text = f"{title} {snippet}"
        
        # Check if result contains grant-related keywords
        has_grant_keywords = any(keyword in combined_text for keyword in grant_keywords)
        
        # Check if result contains excluded keywords
        has_exclude_keywords = any(keyword in combined_text for keyword in exclude_keywords)
        
        # Filter based on domain reputation (optional)
        link = result.get("link", "")
        is_reputable_domain = is_reputable_grant_domain(link)
        
        if has_grant_keywords and not has_exclude_keywords and is_reputable_domain:
            filtered.append(result)
    
    return filtered

def is_reputable_grant_domain(url):
    """Check if domain is from a reputable source"""
    reputable_domains = [
        "grants.gov", "foundation", "nonprofit", "charity", "edu", "org",
        "gov", "hud.gov", "nih.gov", "nsf.gov", "usda.gov", "epa.gov",
        "fordFoundation", "gatesfoundation", "rockefellerfoundation",
        "guidestar", "candid.org", "councilofnonprofits"
    ]
    
    return any(domain in url.lower() for domain in reputable_domains)

def remove_duplicates(results):
    """Remove duplicate results based on URL"""
    seen_urls = set()
    unique_results = []
    
    for result in results:
        url = result.get("link", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    
    return unique_results

def rank_grant_results(results, query, location):
    """Rank results based on relevance to query and location"""
    
    def calculate_relevance_score(result):
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        combined_text = f"{title} {snippet}"
        
        score = 0
        
        # Query relevance
        query_words = query.lower().split()
        for word in query_words:
            if word in combined_text:
                score += 2
        
        # Location relevance
        if location:
            location_words = location.lower().split()
            for word in location_words:
                if word in combined_text:
                    score += 1
        
        # Boost for certain high-value keywords
        high_value_keywords = ["deadline", "apply", "application", "funding available"]
        for keyword in high_value_keywords:
            if keyword in combined_text:
                score += 3
        
        # Boost for government and foundation sources
        url = result.get("link", "").lower()
        if any(domain in url for domain in ["gov", "foundation", "org"]):
            score += 1
        
        return score
    
    # Sort by relevance score (highest first)
    ranked = sorted(results, key=calculate_relevance_score, reverse=True)
    
    return ranked

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
@app.route("/find-grants", methods=["POST"])
def find_grants():
    data = request.get_json()
    city = data.get("city")
    state = data.get("state")

    # Your logic for looking up grants based on city/state
    results = [
        {"title": "Small Business Boost", "amount": "$10,000", "source": "Illinois DCEO"},
        {"title": "Community Development Block Grant", "amount": "$25,000", "source": "City of Chicago"},
    ]

    return render_template("grants_results.html", results=results, city=city, state=state)

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
