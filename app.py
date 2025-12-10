import os
import json
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime, timedelta
from dateparser.search import search_dates
from openai import OpenAI
import requests

# --- Setup OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Setup API Endpoint  ---
HOTEL_API_ENDPOINT = os.getenv("HOTEL_API_ENDPOINT")
HOTEL_API_BEARER_TOKEN = os.getenv("HOTEL_API_BEARER_TOKEN")

def get_hotel_availability(json_key: str, start: str, end: str, adults: str, kids: str, minors: str):
    headers = {
        "Authorization": f"Bearer {HOTEL_API_BEARER_TOKEN}",
        "Accept": "*/*",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0"
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
        response = requests.get(HOTEL_API_ENDPOINT, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        print("\n--- Hotel Availability API Call ---")
        print("Request URL:", response.url)
        print("Request Headers:", headers)
        print("Request Params:", params)
        print("Response Data:", json.dumps(data, indent=2))
        print("--- End of API Call ---\n")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error calling hotel availability API: {e}")
        return {"error": "API call failed", "details": str(e)}

def get_hotel_price(json_key: str, start: str, end: str, adults: str, kids: str, minors: str):
    headers = {
        "Authorization": f"Bearer {HOTEL_API_BEARER_TOKEN}",
        "Accept": "*/*",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0"
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
        response = requests.get(HOTEL_API_ENDPOINT, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        print("\n--- Hotel Price API Call ---")
        print("Request URL:", response.url)
        print("Request Headers:", headers)
        print("Request Params:", params)
        print("Response Data:", json.dumps(data, indent=2))
        print("--- End of API Call ---\n")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error calling hotel price API: {e}")
        return {"error": "API call failed", "details": str(e)}

# --- Tools definitions ---
tool_availability = {
    "name": "get_hotel_availability",
    "description": "Get hotel room availability for a given date range.",
    "parameters": {
        "type": "object",
        "properties": {
            "json_key": {"type": "string", "description": "The key for the JSON response, e.g., 'availability'"},
            "start": {"type": "string", "description": "Start date YYYY-MM-DD"},
            "end":   {"type": "string", "description": "End date YYYY-MM-DD"},
            "adults":   {"type": "string", "description": "The total number of adult guests"},
            "kids":   {"type": "string", "description": "The total number of children between 3 and 12 years old"},
            "minors":   {"type": "string", "description": "The total number of children guests below 3 years old"}
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
            "json_key": {"type": "string", "description": "Must be 'price'"},
            "start": {"type": "string", "description": "Start date YYYY-MM-DD"},
            "end":   {"type": "string", "description": "End date YYYY-MM-DD"},
            "adults":   {"type": "string", "description": "The total number of adult guests"},
            "kids":   {"type": "string", "description": "The total number of children between 3 and 12 years old"},
            "minors":   {"type": "string", "description": "The total number of children guests below 3 years old"}
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

messages = [{"role": "system", "content": system_prompt}]

def try_handle_tool_call(ai_message):
    fc = ai_message.function_call
    if not fc:
        return None

    if fc.name == "get_hotel_availability":
        args = json.loads(fc.arguments or "{}")
        result = get_hotel_availability(**args)
        return {
            "role": "function",
            "name": fc.name,
            "content": json.dumps(result)
        }

    if fc.name == "get_hotel_price":
        args = json.loads(fc.arguments or "{}")
        result = get_hotel_price(**args)
        return {
            "role": "function",
            "name": fc.name,
            "content": json.dumps(result)
        }

    return None

def call_gpt(messages, tools=None):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=(tools if tools else []),
        function_call="auto"
    )
    return resp.choices[0].message

# --- UI setup ---
output = widgets.Output()
input_box = widgets.Text(placeholder="Type your message here...")
send_button = widgets.Button(description="Send", button_style="success")

def on_send(_):
    user_msg = input_box.value.strip()
    if not user_msg:
        return
    with output:
        print(f"User: {user_msg}")
    messages.append({"role": "user", "content": user_msg})

    ai_msg = call_gpt(messages, tools=[tool_availability, tool_price])
    messages.append({
        "role": "assistant",
        "content": ai_msg.content,
        "function_call": ai_msg.function_call
    })

    tool_response = try_handle_tool_call(ai_msg)
    if tool_response:
        messages.append(tool_response)
        ai_msg2 = call_gpt(messages, tools=[tool_availability, tool_price])
        messages.append({
            "role": "assistant",
            "content": ai_msg2.content,
            "function_call": ai_msg2.function_call
        })
        with output:
            print(f"Bot: {ai_msg2.content}")
    else:
        with output:
            print(f"Bot: {ai_msg.content}")

    input_box.value = ""

send_button.on_click(on_send)
input_box.on_submit(lambda _: on_send(None))
display(widgets.VBox([output, widgets.HBox([input_box, send_button])]))
