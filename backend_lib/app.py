import os
import json
from datetime import datetime, timedelta
from dateparser.search import search_dates
from openai import OpenAI
import requests

# --- OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
HOTEL_API_BEARER_TOKEN = os.getenv("HOTEL_API_BEARER_TOKEN")

# --- Hotel API functions ---
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

# --- Tool definitions ---
tool_availability = {
    "name": "get_hotel_availability",
    "description": "Get hotel room availability for a given date range.",
    "parameters": {
        "type": "object",
        "properties": {
            "json_key": {"type": "string"},
            "start": {"type": "string"},
            "end":   {"type": "string"},
            "adults": {"type": "string"},
            "kids": {"type": "string"},
            "minors": {"type": "string"}
        },
        "required": ["json_key", "start", "end", "adults", "kids", "minors"]
    }
}

tool_price = {
    "name": "get_hotel_price",
    "description": "Get hotel room prices for a given date range.",
    "parameters": {
        "type": "object",
        "properties": {
            "json_key": {"type": "string"},
            "start": {"type": "string"},
            "end":   {"type": "string"},
            "adults": {"type": "string"},
            "kids": {"type": "string"},
            "minors": {"type": "string"}
        },
        "required": ["json_key", "start", "end", "adults", "kids", "minors"]
    }
}

# --- System prompt ---
system_prompt = """
You are a hotel reception assistant for Hotel Ilion.
You can chat naturally with the user, ask clarifying questions, and only call the get_hotel_availability function when you are *sure* the user has provided the check-in date, check-out date, and number of guests.
Never reveal available rooms number, room names, room types, or room numbers.
Always clarify ages of children.
When calling the function, respond with a JSON function call. Otherwise respond naturally and always with the same language as the input.
"""

# --- Chat state ---
messages = [{"role": "system", "content": system_prompt}]

# --- Tool call handler ---
def try_handle_tool_call(ai_message):
    fc = ai_message.function_call
    if not fc:
        return None
    if fc.name == "get_hotel_availability":
        args = json.loads(fc.arguments or "{}")
        result = get_hotel_availability(**args)
        return {"role": "function", "name": fc.name, "content": json.dumps(result)}
    if fc.name == "get_hotel_price":
        args = json.loads(fc.arguments or "{}")
        result = get_hotel_price(**args)
        return {"role": "function", "name": fc.name, "content": json.dumps(result)}
    return None

# --- GPT call wrapper ---
def call_gpt(messages, tools=None):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=(tools if tools else []),
        function_call="auto"
    )
    return resp.choices[0].message
