import os
import json
from datetime import datetime
from dateparser.search import search_dates
from openai import OpenAI
import requests

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
HOTEL_API_BEARER_TOKEN = os.getenv("HOTEL_API_BEARER_TOKEN")

# Hotel availability API
def get_hotel_availability(json_key: str, start: str, end: str, adults: str, kids: str, minors: str):
    headers = {
        "Authorization": f"Bearer {HOTEL_API_BEARER_TOKEN}",
        "Accept": "*/*",
        "User-Agent": "Mozilla/5.0"
    }
    params = {
        "json_key": json_key,
        "start": start,
        "end": end,
        "adults": adults,
        "kids": kids,
        "minors": minors
    }
    try:
        response = requests.get("https://hotel.dev-maister.gr/hotel_Casa/mcp_server/index.php",
                                headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": "API call failed", "details": str(e)}

# Hotel price API
def get_hotel_price(json_key: str, start: str, end: str, adults: str, kids: str, minors: str):
    headers = {
        "Authorization": f"Bearer {HOTEL_API_BEARER_TOKEN}",
        "Accept": "*/*",
        "User-Agent": "Mozilla/5.0"
    }
    params = {
        "json_key": json_key,
        "start": start,
        "end": end,
        "adults": adults,
        "kids": kids,
        "minors": minors
    }
    try:
        response = requests.get("https://hotel.dev-maister.gr/hotel_Casa/mcp_server/index.php",
                                headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": "API call failed", "details": str(e)}

# Chatbot system
system_prompt = """You are a hotel reception assistant for Hotel Ilion.
You can chat naturally, ask clarifying questions, and only call the get_hotel_availability function
when you are sure the user provided check-in, check-out, and guests info.
Never reveal room numbers or types.
"""

# Messages
messages = [{"role": "system", "content": system_prompt}]

# Tool definitions
tool_availability = { "name": "get_hotel_availability", "description": "...", "parameters": {} }
tool_price = { "name": "get_hotel_price", "description": "...", "parameters": {} }

# Tool call handler
def try_handle_tool_call(ai_message):
    fc = getattr(ai_message, "function_call", None)
    if not fc: return None
    if fc.name == "get_hotel_availability":
        args = json.loads(fc.arguments or "{}")
        return {"role": "function", "name": fc.name, "content": json.dumps(get_hotel_availability(**args))}
    if fc.name == "get_hotel_price":
        args = json.loads(fc.arguments or "{}")
        return {"role": "function", "name": fc.name, "content": json.dumps(get_hotel_price(**args))}
    return None

# GPT call wrapper
def call_gpt(messages, tools=None):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=(tools if tools else []),
        function_call="auto"
    )
    return resp.choices[0].message
