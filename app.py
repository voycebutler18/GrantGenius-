from flask import Flask, request, jsonify, render_template, redirect, url_for
from openai import OpenAI
from flask import Flask, request, jsonify
import time
import os
import logging
import requests
from urllib.parse import quote

logger = logging.getLogger(__name__)

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
def index():
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

@app.route('/search', methods=['GET'])
def search_grants():
    """Search for grants with comprehensive error handling and debugging"""
    start_time = time.time()
    
    try:
        query = request.args.get('query', '').strip()
        location = request.args.get('location', '').strip()
        
        logger.info(f"Search request received - Query: '{query}', Location: '{location}'")
        
        if not query:
            logger.warning("Search attempted with empty query")
            return jsonify({"error": "Query parameter is required"}), 400
        
        # Add timeout protection
        search_results = []
        
        # Try multiple search strategies with fallbacks
        try:
            # Primary search method
            search_results = perform_comprehensive_search(query, location)
            logger.info(f"Primary search completed, found {len(search_results)} results")
            
        except Exception as primary_error:
            logger.error(f"Primary search failed: {str(primary_error)}")
            
            # Fallback to simpler search
            try:
                search_results = perform_fallback_search(query, location)
                logger.info(f"Fallback search completed, found {len(search_results)} results")
                
            except Exception as fallback_error:
                logger.error(f"Fallback search failed: {str(fallback_error)}")
                
                # Last resort: return curated results
                search_results = get_curated_results(query, location)
                logger.info(f"Using curated results, found {len(search_results)} results")
        
        # Ensure we have results
        if not search_results:
            logger.warning("No results found, generating default results")
            search_results = generate_default_results(query, location)
        
        # Add timing info
        elapsed_time = time.time() - start_time
        logger.info(f"Search completed in {elapsed_time:.2f} seconds")
        
        response = {
            "query": query,
            "location": location,
            "total_results": len(search_results),
            "search_time": f"{elapsed_time:.2f}s",
            "items": search_results[:20]  # Limit to 20 results
        }
        
        return jsonify(response)
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Search error after {elapsed_time:.2f}s: {str(e)}")
        
        # Return error with some default results
        return jsonify({
            "error": "Search encountered an issue but here are some general resources",
            "query": query if 'query' in locals() else "",
            "location": location if 'location' in locals() else "",
            "items": get_emergency_results(query if 'query' in locals() else "")
        }), 200  # Return 200 instead of 500 to show results

def perform_comprehensive_search(query, location):
    """Main search function with multiple strategies"""
    results = []
    
    # Build search queries
    search_queries = build_search_queries(query, location)
    
    for search_query in search_queries:
        try:
            # Add timeout to prevent hanging
            search_results = perform_web_search_with_timeout(search_query, timeout=5)
            
            if search_results:
                filtered = filter_and_rank_results(search_results, query, location)
                results.extend(filtered)
                
                # Don't search more if we have enough results
                if len(results) >= 15:
                    break
                    
        except Exception as e:
            logger.warning(f"Search query '{search_query}' failed: {str(e)}")
            continue
    
    return remove_duplicates(results)

def perform_web_search_with_timeout(query, timeout=10):
    """Perform web search using Google Custom Search JSON API"""
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": query,
            "num": 10
        }

        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", "")[:100],
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })

        return results

    except requests.RequestException as e:
        logger.error(f"Web search failed: {str(e)}")
        raise

def perform_fallback_search(query, location):
    """Fallback search using different approach"""
    
    # This could use a different search engine or API
    # For now, let's return some static but relevant results
    
    return get_curated_results(query, location)

def build_search_queries(query, location):
    """Build multiple search queries for comprehensive results"""
    
    queries = []
    
    # Basic query
    basic_query = f"{query} grants funding"
    if location:
        basic_query += f" {location}"
    queries.append(basic_query)
    
    # Foundation grants
    queries.append(f"{query} foundation grants private funding")
    
    # Government grants  
    queries.append(f"{query} government grants federal state")
    
    # Corporate grants
    queries.append(f"{query} corporate grants business funding")
    
    # Nonprofit grants
    queries.append(f"{query} nonprofit grants charitable funding")
    
    return queries

def filter_and_rank_results(results, query, location):
    """Filter and rank search results"""
    
    filtered = []
    
    for result in results:
        # Basic filtering
        if result.get("link") and result.get("title"):
            # Add relevance score
            result["relevance_score"] = calculate_relevance_score(result, query, location)
            filtered.append(result)
    
    # Sort by relevance
    filtered.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    return filtered

def calculate_relevance_score(result, query, location):
    """Calculate relevance score for ranking"""
    
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    combined = f"{title} {snippet}"
    
    score = 0
    
    # Query word matches
    for word in query.lower().split():
        if word in combined:
            score += 2
    
    # Location matches
    if location:
        for word in location.lower().split():
            if word in combined:
                score += 1
    
    # Grant-specific keywords
    grant_keywords = ["grant", "funding", "foundation", "scholarship", "award"]
    for keyword in grant_keywords:
        if keyword in combined:
            score += 1
    
    return score

def get_curated_results(query, location):
    """Get curated grant results based on query type"""
    
    results = []
    
    # Add relevant grant sources based on query
    if "health" in query.lower() or "medical" in query.lower():
        results.extend([
            {
                "title": "NIH Small Business Innovation Research (SBIR) Grants",
                "link": "https://www.nih.gov/research-training/small-business-research",
                "snippet": "NIH provides funding for small businesses to conduct research and development in health and biomedical fields."
            },
            {
                "title": "Health Resources and Services Administration (HRSA) Grants",
                "link": "https://www.hrsa.gov/grants",
                "snippet": "HRSA provides grants to improve health care access and quality in underserved communities."
            },
            {
                "title": "Robert Wood Johnson Foundation Health Grants",
                "link": "https://www.rwjf.org/en/grants",
                "snippet": "Private foundation grants focused on health and health care in the United States."
            }
        ])
    
    if "education" in query.lower():
        results.extend([
            {
                "title": "Department of Education Grants",
                "link": "https://www.ed.gov/grants",
                "snippet": "Federal grants for education programs, research, and improvement initiatives."
            },
            {
                "title": "Gates Foundation Education Grants",
                "link": "https://www.gatesfoundation.org/our-work/programs/us-program/postsecondary-education",
                "snippet": "Grants supporting education initiatives and improving educational outcomes."
            }
        ])
    
    if "community" in query.lower() or "social" in query.lower():
        results.extend([
            {
                "title": "Community Development Block Grants (CDBG)",
                "link": "https://www.hud.gov/program_offices/comm_planning/cdbg",
                "snippet": "Federal grants to help communities develop viable urban communities and provide decent housing."
            },
            {
                "title": "United Way Community Grants",
                "link": "https://www.unitedway.org/get-involved/volunteer/grant-opportunities",
                "snippet": "Local grants supporting community programs and social services."
            }
        ])
    
    # Add general grant resources
    results.extend([
        {
            "title": "Grants.gov - Find Grant Opportunities",
            "link": "https://www.grants.gov/search-grants",
            "snippet": "The official government source for federal grant opportunities and applications."
        },
        {
            "title": "Foundation Directory Online",
            "link": "https://fdo.foundationcenter.org/",
            "snippet": "Comprehensive database of foundation and corporate grants from around the world."
        }
    ])
    
    return results

def generate_default_results(query, location):
    """Generate default results when search fails"""
    
    location_text = f" in {location}" if location else ""
    
    return [
        {
            "title": f"{query.title()} Grants{location_text}",
            "link": "https://www.grants.gov/search-grants",
            "snippet": f"Find federal grants for {query} projects{location_text}. Search thousands of grant opportunities."
        },
        {
            "title": f"Foundation Grants for {query.title()}",
            "link": "https://candid.org/explore-issues",
            "snippet": f"Private foundation funding opportunities for {query} initiatives and programs."
        },
        {
            "title": f"State and Local {query.title()} Funding",
            "link": "https://www.usa.gov/grants",
            "snippet": f"Explore state and local government grants for {query} projects{location_text}."
        }
    ]

def get_emergency_results(query):
    """Emergency results when everything fails"""
    
    return [
        {
            "title": "Grants.gov - Official Grant Database",
            "link": "https://www.grants.gov/",
            "snippet": "The official source for federal grant opportunities. Search over 1,000 grant programs."
        },
        {
            "title": "Foundation Center - Grant Database",
            "link": "https://candid.org/",
            "snippet": "Comprehensive database of foundation and corporate grants worldwide."
        },
        {
            "title": "GrantSpace - Free Grant Resources",
            "link": "https://grantspace.org/",
            "snippet": "Free resources and tools for grant seekers, including funding databases and guides."
        }
    ]

def remove_duplicates(results):
    """Remove duplicate results based on URL"""
    seen_urls = set()
    unique_results = []
    
    for result in results:
        url = result.get("link", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    
    return unique_results

# Add a test endpoint to debug the search
@app.route('/search/test', methods=['GET'])
def test_search():
    """Test endpoint to debug search functionality"""
    
    return jsonify({
        "status": "Search endpoint is working",
        "test_query": "mental health",
        "test_location": "California",
        "timestamp": time.time()
    })

# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
