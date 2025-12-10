import os
import json
from datetime import datetime
from dateparser.search import search_dates
from openai import OpenAI
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
HOTEL_API_BEARER_TOKEN = os.getenv("HOTEL_API_BEARER_TOKEN")

# --- GPT / Hotel API logic ---
system_prompt = """You are a hotel reception assistant for Hotel Ilion.
You can chat naturally and only call the hotel API when you have dates and guests.
Never reveal room numbers or room types.
"""

messages = [{"role": "system", "content": system_prompt}]

# --- Hotel API functions ---
def get_hotel_availability(json_key: str, start: str, end: str, adults: str, kids: str, minors: str):
    headers = {
        "Authorization": f"Bearer {HOTEL_API_BEARER_TOKEN}",
        "Accept": "*/*",
        "User-Agent": "Mozilla/5.0"
    }
    params = {"json_key": json_key, "start": start, "end": end, "adults": adults, "kids": kids, "minors": minors}
    try:
        response = requests.get(
            "https://hotel.dev-maister.gr/hotel_Casa/mcp_server/index.php",
            headers=headers, params=params, timeout=10
        )
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
    params = {"json_key": json_key, "start": start, "end": end, "adults": adults, "kids": kids, "minors": minors}
    try:
        response = requests.get(
            "https://hotel.dev-maister.gr/hotel_Casa/mcp_server/index.php",
            headers=headers, params=params, timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": "API call failed", "details": str(e)}

# --- Tools definitions ---
tool_availability = {
    "name": "get_hotel_availability",
    "description": "Get hotel room availability for a given date range.",
    "parameters": {}
}

tool_price = {
    "name": "get_hotel_price",
    "description": "Get hotel room prices for a given date range.",
    "parameters": {}
}

# --- Tool call handler ---
def try_handle_tool_call(ai_message):
    fc = getattr(ai_message, "function_call", None)
    if not fc:
        return None
    if fc.name == "get_hotel_availability":
        args = json.loads(fc.arguments or "{}")
        return {"role": "function", "name": fc.name, "content": json.dumps(get_hotel_availability(**args))}
    if fc.name == "get_hotel_price":
        args = json.loads(fc.arguments or "{}")
        return {"role": "function", "name": fc.name, "content": json.dumps(get_hotel_price(**args))}
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

# --- FastAPI app ---
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request model ---
class UserMessage(BaseModel):
    message: str

# --- Chat endpoint ---
@app.post("/chat")
async def chat(msg: UserMessage):
    user_msg = msg.message.strip()
    if not user_msg:
        return {"error": "Empty message"}

    # Append user message
    messages.append({"role": "user", "content": user_msg})

    # GPT call
    ai_msg = call_gpt(messages, tools=[tool_availability, tool_price])
    messages.append({
        "role": "assistant",
        "content": ai_msg.content,
        "function_call": ai_msg.function_call
    })

    # Handle tool call if any
    tool_response = try_handle_tool_call(ai_msg)
    if tool_response:
        messages.append(tool_response)
        ai_msg2 = call_gpt(messages, tools=[tool_availability, tool_price])
        messages.append({
            "role": "assistant",
            "content": ai_msg2.content,
            "function_call": ai_msg2.function_call
        })
        return {"reply": ai_msg2.content}

    return {"reply": ai_msg.content}

# --- This line is crucial for Vercel ---
handler = app
