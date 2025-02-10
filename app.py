from fastapi import FastAPI, Request, Form, Body, HTTPException, Depends, Cookie
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import google.generativeai as genai
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import openai
import socket
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
import firebase_admin
from firebase_admin import credentials, auth
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import json
import certifi


class ApiKeys(BaseModel):
    selectedModel: str
    geminiApiKey: Optional[str] = None
    geminiModel: Optional[str] = None
    chatgptApiKey: Optional[str] = None
    deepseekApiKey: Optional[str] = None

class GenerateRequest(BaseModel):
    input: str
    prompt: str
    section: str
    model: str
    geminiApiKey: Optional[str] = None
    chatgptKey: Optional[str] = None
    chatgptModel: Optional[str] = None
    deepseekApiKey: Optional[str] = None
    deepseekModel: Optional[str] = None

# Initialize Firebase Admin with your service account
cred = credentials.Certificate("firebase_key.json")  # Make sure this path is correct
firebase_admin.initialize_app(cred)

security = HTTPBearer()

# Configure templates and static files
templates = Jinja2Templates(directory="templates")

# Load environment variables
load_dotenv()

# Update API keys to use environment variables
SERPAPI_KEY = os.getenv('SERPAPI_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize FastAPI with all configurations
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
mongodb_client = None

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Update Chrome options for Render's environment
def get_chrome_options():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    return chrome_options

# Update webdriver initialization
def get_webdriver():
    try:
        chrome_options = get_chrome_options()
        service = ChromeService()
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        print(f"Error initializing webdriver: {e}")
        return None

def scrape_website(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def get_html_tags_info(html_content):
    result = analyze_html(html_content, keyword = "Abstract")
    if result == []:
        result = analyze_html(html_content, keyword = "Summary")
    return result


def analyze_html(html_content, keyword):
    soup = BeautifulSoup(html_content, 'html.parser')
    # keyword_count = soup.get_text().lower().count(keyword.lower())
    # if keyword_count >= 4:
    #     return f"The keyword '{keyword}' appears {keyword_count} times, which is too frequent."

    tags_to_check = ['h1', 'h2', 'h3', 'p']
    extracted_data = []

    for tag_name in tags_to_check:
        for tag in soup.find_all(tag_name):
            if keyword.lower() in tag.get_text().lower():
                tag_text = tag.get_text(strip=True)
                if len(tag_text) <= len(keyword) + 5:
                    parent_tag_text = tag.find_parent().get_text(strip=True)
                    lvl_counter = 1
                    while (parent_tag_text == tag_text) and lvl_counter<=3:
                        new_tag = tag.find_parent()
                        parent_tag_text = new_tag.find_parent().get_text(strip=True)
                        lvl_counter+=1
                    extracted_data.append({
                        'tag': tag_name,
                        'tag_text': tag_text,
                        'parent_tag_text': parent_tag_text
                    })

    return extracted_data if extracted_data else []

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
    
def get_webdriver(): # removed global variable to check if it works fine, as earlier it is failing.
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    free_port = find_free_port()
    chrome_options.add_argument(f'--remote-debugging-port={free_port}')
    SELENIUM_DRIVER = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    return SELENIUM_DRIVER

def get_html_content(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/103.0.0.0 Safari/537.36"
        }
        temp = requests.get(url, headers=headers, timeout=10)
        status_code = temp.status_code
        if status_code==200:
            html_content = temp.text
            return html_content, status_code, 'Request'
        
        # print(status_code, url)
        driver = get_webdriver()
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        html_content = driver.find_element(By.TAG_NAME, 'html').get_attribute('innerHTML')
        driver.quit()
        return html_content, 200, 'Selenium'
    except Exception as e:
        return "", 500, str(e)

async def get_gemini_search_strings(text, model):
    prompt = f"""
        Act as a PhD student conducting research for their thesis. The thesis topic is as follows: {text}.
        Analyze the core research problem, underlying themes, and critical aspects of the thesis.
        Based on this analysis, generate highly effective and specific search strings that capture the essence and crux of the thesis.
        The search strings should focus on the conceptual foundation, key challenges, and significant debates related to the topic, rather than just its keywords.
        Ensure the search strings maximize the relevance of the results to the thesis and lead to high-quality academic sources.
        """
    # Each search string should include specific sites (e.g., Google Scholar, PubMed, ResearchGate, or relevant university repositories) to ensure high-quality results.
    resp = model.generate_content(prompt)
    return resp.text

async def generate_with_gemini(prompt: str, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text if response else ""

async def generate_with_chatgpt(prompt: str, api_key: str, model_name: str) -> str:
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful research paper assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content if response.choices else ""

async def generate_with_deepseek(prompt: str, api_key: str, model_name: str) -> str:
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful research paper assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise HTTPException(status_code=response.status_code, detail="DeepSeek API error")
    except Exception as e:
        print(f"DeepSeek API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Authentication dependency
async def verify_token_from_cookie(request: Request):
    try:
        token = request.cookies.get("auth_token")
        if not token:
            raise HTTPException(status_code=403, detail="Not authenticated")
        
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=403, detail="Not authenticated")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        await verify_token_from_cookie(request)
        return RedirectResponse(url='/app')
    except HTTPException:
        return templates.TemplateResponse("login.html", {"request": request})

@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    try:
        await verify_token_from_cookie(request)
        return templates.TemplateResponse("index.html", {"request": request})
    except HTTPException:
        return RedirectResponse(url='/')

@app.post("/verify-token")
async def verify_firebase_token(data: dict = Body(...)):
    try:
        token = data.get("token")
        if not token:
            raise HTTPException(status_code=400, detail="No token provided")
        
        decoded_token = auth.verify_id_token(token)
        
        # Create a JSON response with the cookie
        response = JSONResponse(content={"status": "success", "uid": decoded_token["uid"]})
        
        # Set the cookie in the response
        response.set_cookie(
            key="auth_token",
            value=token,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite='lax',
            max_age=3600  # 1 hour
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/generate")
async def generate_content(
    request: Request,
    data: dict = Body(...),
):
    user_info = await verify_token_from_cookie(request)
    try:
        print(f"Processing request for section: {data['section']} with model: {data['model']}")
        
        # Get the combined input and initialize combined output
        combined_input = data.get("input", "")
        combined_output = ""
        current_section = data.get("section", "")
        
        if data['model'] == "gemini":
            if not data['geminiApiKey']:
                raise HTTPException(status_code=400, detail="Gemini API key is required")
            content = await generate_with_gemini(data['prompt'], data['geminiApiKey'])
            # Format the section output
            combined_output = f"### {current_section}\n{content}"
        
        elif data['model'] == "chatgpt":
            if not data['chatgptKey'] or not data['chatgptModel']:
                raise HTTPException(status_code=400, detail="ChatGPT API key and model are required")
            content = await generate_with_chatgpt(data['prompt'], data['chatgptKey'], data['chatgptModel'])
            combined_output = f"### {current_section}\n{content}"
        
        elif data['model'] == "deepseek":
            if not data['deepseekApiKey'] or not data['deepseekModel']:
                raise HTTPException(status_code=400, detail="DeepSeek API key and model are required")
            content = await generate_with_deepseek(data['prompt'], data['deepseekApiKey'], data['deepseekModel'])
            combined_output = f"### {current_section}\n{content}"
        
        else:
            raise HTTPException(status_code=400, detail="Invalid model selection")

        if content:
            try:
                print("Saving to chat history...")
                
                # Check if there's an existing document for this input
                existing_doc = await app.mongodb.chat_history.find_one({
                    "user_id": user_info["uid"],
                    "input_text": combined_input,
                    "timestamp": {
                        "$gte": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                    }
                })

                if existing_doc:
                    # Append the new section to existing output
                    updated_output = existing_doc["output_text"] + "\n\n" + combined_output
                    await app.mongodb.chat_history.update_one(
                        {"_id": existing_doc["_id"]},
                        {"$set": {"output_text": updated_output}}
                    )
                else:
                    # Create new document
                    await save_chat_history(
                        user_id=user_info["uid"],
                        input_text=combined_input,
                        output_text=combined_output,
                        model=data['model']
                    )
                
                print("Successfully saved to chat history")
                
                return {
                    "status": "success",
                    "content": content,
                    "section": data['section']
                }
            except Exception as db_error:
                print(f"Error saving to MongoDB: {str(db_error)}")
                # Continue with the response even if saving to history fails
                return {
                    "status": "success",
                    "content": content,
                    "section": data['section']
                }
        else:
            raise HTTPException(status_code=500, detail="Empty response from AI model")
            
    except Exception as e:
        print(f"Error in generate_content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_research(
    request: Request,
    data: dict = Body(...),
):
    await verify_token_from_cookie(request)
    try:
        # Extract data from request body
        query = data.get('query')
        num_websites = data.get('numWebsites', 5)  # Default to 5 websites
        content_limit = data.get('contentLimit', 1000)  # Default to 1000 characters
        model_type = data.get('model')
        gemini_api_key = data.get('geminiApiKey')
        chatgpt_api_key = data.get('chatgptApiKey')
        chatgpt_model = data.get('chatgptModel')

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Configure APIs with provided keys

        if not gemini_api_key:
            raise HTTPException(status_code=400, detail="Gemini API key is required")
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')

        # Get search strings using the selected model
        search_strings = await get_gemini_search_strings(query, model) if model_type == 'gemini' else query

        # Use SerpAPI to get search results
        params = {
            "engine": "google",
            "q": search_strings + " research paper",
            "api_key": '4a3af6c37b5620616e93976ff767ad2ff0373421e21d715468dc9dce9b377184',
            "num": int(num_websites)
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Extract URLs from organic results
        search_results = []
        if "organic_results" in results:
            search_results = [result["link"] for result in results["organic_results"][:int(num_websites)]]
        
        # Scrape content from each URL
        all_content = []
        url_content_parsed = {}
        for url in search_results:
            if url.endswith('.pdf'):
                continue
            content, status_code, source = get_html_content(url)
            if content:
                tags_info = get_html_tags_info(content)
                url_content_parsed[url] = {
                    'url': url,
                    'source': source,
                    'tags_info': tags_info,
                    'status_code': status_code
                }

            if tags_info:  # Only add if we got meaningful content
                tags_info = "\n".join([tag['tag_text'] for tag in tags_info])
                all_content.append(f"Source: {url}\nContent Summary:\n{tags_info}")
        
        # Combine all content with character limit
        combined_content = "\n\n".join(all_content)
        if content_limit:
            combined_content = combined_content[:int(content_limit)]

        return {
            "search string used": search_strings,
            "status": "success",
            "results": combined_content,
            "sources": search_results,
            "url_content_parsed": url_content_parsed
        }

    except Exception as e:
        print(f"Error in search_research: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Add logout endpoint
@app.post("/logout")
async def logout():
    response = JSONResponse(content={"status": "success"})
    response.delete_cookie("auth_token")
    return response

@app.on_event("startup")
async def startup_db_client():
    global mongodb_client
    print("Initializing MongoDB connection...")
    mongodb_client = AsyncIOMotorClient(
        os.getenv("MONGODB_URI"),
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=5000
    )
    app.mongodb = mongodb_client.hkare_ai

    # Test the connection
    try:
        await mongodb_client.admin.command('ping')
        print("Successfully connected to MongoDB")
        # Test database and collection
        await app.mongodb.chat_history.find_one()
        print("Successfully accessed chat_history collection")
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_db_client():
    if mongodb_client:
        mongodb_client.close()

# Add this function to save chat history
async def save_chat_history(
    user_id: str, 
    input_text: str, 
    output_text: str, 
    model: str, 
):
    try:
        print(f"Preparing history document for user {user_id}")
        history_doc = {
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "input_text": input_text,
            "output_text": output_text,
            "model_used": model,
        }
        print("Inserting document into MongoDB...")
        await app.mongodb.chat_history.insert_one(history_doc)
        print("Successfully inserted document")
    except Exception as e:
        print(f"Error in save_chat_history: {str(e)}")
        raise e

# Update the history endpoint to fetch chat history
@app.get("/history")
async def get_history(request: Request):
    user_info = await verify_token_from_cookie(request)
    try:
        cursor = app.mongodb.chat_history.find(
            {"user_id": user_info["uid"]},
            {"_id": 0}
        ).sort("timestamp", 1)  # Sort by oldest first
        
        history = await cursor.to_list(length=None)  # Get all entries
        
        # Convert datetime objects to strings
        for entry in history:
            entry["timestamp"] = entry["timestamp"].isoformat()
        
        return {"status": "success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add this new endpoint
@app.post("/delete_history")
async def delete_history_entry(
    request: Request,
    data: dict = Body(...),
):
    user_info = await verify_token_from_cookie(request)
    try:
        # Convert timestamp string to datetime
        timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        
        # Delete the entry
        result = await app.mongodb.chat_history.delete_one({
            "user_id": user_info["uid"],
            "timestamp": timestamp,
            "input_text": data['input_text']
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Entry not found")
            
        return {"status": "success", "message": "Entry deleted successfully"}
    except Exception as e:
        print("Error deleting history entry:", str(e))
        raise HTTPException(status_code=500, detail=str(e)) 