import streamlit as st
import pandas as pd
import requests
import time
import json
import re
import zipfile
import base64
from io import BytesIO
from docx import Document
from huggingface_hub import InferenceClient

# Configure page
st.set_page_config(page_title="SEO Content Automation", page_icon="📚", layout="wide")

# Initialize session state
if "articles" not in st.session_state:
    st.session_state["articles"] = {}
if "images" not in st.session_state:
    st.session_state["images"] = {}
if "publish_log" not in st.session_state:
    st.session_state["publish_log"] = []

def init_hf_client():
    """Initialize Hugging Face client"""
    try:
        HF_TOKEN = st.secrets.get("HF_TOKEN")
        if not HF_TOKEN:
            return None
        return InferenceClient(
            model="stabilityai/stable-diffusion-3-medium",
            token=HF_TOKEN
        )
    except Exception as e:
        st.error(f"Error initializing HF client: {str(e)}")
        return None

def get_api_key(provider):
    """Get API key for selected provider"""
    if provider == "Grok (X.AI)":
        return st.secrets.get("GROK_API_KEY")
    elif provider == "OpenAI":
        return st.secrets.get("OPENAI_API_KEY")
    return None

def call_ai_for_metadata(keyword, intent, content_type, notes, api_key, provider):
    """Generate metadata using AI API"""
    prompt = f"""
You are a search volume estimator and content strategist for Indian students and professionals.

Given the keyword: "{keyword}"
Intent: {intent}
Content Type: {content_type}
Notes: {notes}

Tasks:
1. Estimate search volume as High, Medium, or Low based on Indian market
2. Create a high-CTR SEO title optimized for Indian audience (include year 2024/2025)
3. Generate a detailed outline with 5-7 bullet points including:
   - Introduction/What is section
   - Key features/benefits
   - Detailed information (eligibility, process, etc.)
   - Tables section (comparison/data)
   - FAQ section
   - Conclusion
4. Provide specific content instructions mentioning:
   - Target keyword usage frequency
   - Required tables (comparison, fees, eligibility, etc.)
   - FAQ requirements (5-8 questions)
   - Tone and style preferences
   - Indian context requirements

Respond in JSON format:
{{
  "volume": "High/Medium/Low",
  "seo_title": "Complete Guide to [Keyword] in India 2024 - Benefits, Process & FAQs",
  "outline": [
    "Introduction - What is [Keyword]?",
    "Key Features and Benefits of [Keyword]", 
    "Detailed [Keyword] Information and Requirements",
    "Comparison Table - [Keyword] Categories/Types",
    "Step-by-Step Application Process",
    "Eligibility Criteria and Documents Required",
    "Frequently Asked Questions (FAQs)",
    "Conclusion and Key Takeaways"
  ],
  "instructions": "Use keyword '[keyword]' 8-12 times naturally. Include 2 detailed tables: one for comparison/categories and one for fees/requirements. Add comprehensive FAQ section with 6-8 questions. Focus on Indian context with specific data. Use professional, informative tone. Include current 2024/2025 information."
}}
"""
    
    # Configure API based on provider
    if provider == "Grok (X.AI)":
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
    elif provider == "OpenAI":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
    else:
        st.error(f"Unsupported provider: {provider}")
        return None

    try:
        response = requests.post(url, json=body, headers=headers)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            try:
                json_data = json.loads(content)
                return json_data
            except:
                return {
                    "volume": "Medium",
                    "seo_title": f"Complete Guide to {keyword} in India 2024 - Benefits, Eligibility & Process",
                    "outline": [
                        f"What is {keyword}? - Introduction and Overview",
                        f"Key Features and Benefits of {keyword}",
                        f"Eligibility Criteria for {keyword}",
                        f"Application Process and Required Documents",
                        f"Detailed Information Table - {keyword} Categories",
                        f"Frequently Asked Questions (FAQs) about {keyword}",
                        "Conclusion and Important Points"
                    ],
                    "instructions": f"Use '{keyword}' naturally 10-15 times. Include 2 tables: eligibility criteria and comparison table. Add FAQ section with 6-8 questions. Focus on Indian context with current data. Professional tone."
                }
        else:
            st.error(f"API Error ({provider}): {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed ({provider}): {str(e)}")
        return None

def generate_article(keyword, seo_title, outline, instructions, api_key, provider):
    """Generate complete article based on metadata"""
    outline_text = "\n".join([f"- {point}" for point in outline])
    
    prompt = f"""
You are an expert content writer specializing in educational content for Indian students and professionals.

Write a complete SEO-optimized article for:
Keyword: "{keyword}"
Title: "{seo_title}"

Structure using this outline:
{outline_text}

Special Instructions:
{instructions}

CRITICAL FORMATTING REQUIREMENTS:
1. Use the target keyword "{keyword}" naturally throughout the article (aim for 1-2% keyword density)
2. Include keyword variations and related terms
3. Structure with clear H1, H2, H3 headings using HTML tags
4. Include at least 1-2 detailed tables with relevant data
5. Add a comprehensive FAQ section at the end with 5-8 questions
6. Use bullet points and numbered lists where appropriate
7. Include specific data, statistics, and numbers
8. Write in HTML format with proper semantic tags

ARTICLE STRUCTURE TEMPLATE:
<h1>{seo_title}</h1>
<p>Introduction paragraph mentioning "{keyword}" and its importance...</p>

<h2>What is {keyword}?</h2>
<p>Detailed explanation...</p>

<h2>Key Features/Benefits of {keyword}</h2>
<ul>
<li>Feature 1 with explanation</li>
<li>Feature 2 with explanation</li>
</ul>

<h2>Detailed Information Table</h2>
<table border="1" cellpadding="8" cellspacing="0">
<tr><th>Parameter</th><th>Details</th></tr>
<tr><td>...</td><td>...</td></tr>
</table>

<h2>How to Apply/Process</h2>
<ol>
<li>Step 1</li>
<li>Step 2</li>
</ol>

<h2>Eligibility Criteria</h2>
<p>Detailed eligibility information...</p>

<h2>Frequently Asked Questions (FAQs)</h2>
<h3>Q1: What is {keyword}?</h3>
<p>A: Detailed answer...</p>

<h3>Q2: Who is eligible for {keyword}?</h3>
<p>A: Detailed answer...</p>

[Continue with 5-8 FAQs total]

<h2>Conclusion</h2>
<p>Summary mentioning "{keyword}" and key takeaways...</p>

Requirements:
- 1000-1500 words
- Natural keyword usage throughout
- Include specific Indian context and data
- Use tables for complex information
- Comprehensive FAQ section
- Professional, informative tone

Write the complete article now following this structure:
"""
    
    # Configure API based on provider
    if provider == "Grok (X.AI)":
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2000
        }
    elif provider == "OpenAI":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2000
        }

    try:
        response = requests.post(url, json=body, headers=headers)
        if response.status_code == 200:
            article = response.json()["choices"][0]["message"]["content"]
            return article
        else:
            st.error(f"Article generation failed ({provider}): {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Article generation error ({provider}): {str(e)}")
        return None

def apply_internal_links(article_content, anchor_map):
    """Apply internal links to article content"""
    linked_article = article_content
    
    # Replace first occurrence of each anchor (case-insensitive)
    for anchor, url in anchor_map.items():
        pattern = re.compile(rf"\b({re.escape(anchor)})\b", re.IGNORECASE)
        linked_article, n = pattern.subn(
            rf'<a href="{url}" target="_blank">\1</a>', 
            linked_article, 
            count=1
        )
    
    # Add related links table
    table_rows = "".join([
        f'<tr><td>{anchor}</td><td><a href="{url}" target="_blank">{url}</a></td></tr>'
        for anchor, url in anchor_map.items()
    ])
    
    link_table = f"""
<div style="margin-top: 30px;">
<h2>
