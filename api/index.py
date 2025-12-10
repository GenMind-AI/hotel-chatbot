from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from hotel_logic import messages, call_gpt, try_handle_tool_call, tool_availability, tool_price

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat(msg: UserMessage):
    user_msg = msg.message.strip()
    if not user_msg:
        return {"error": "Empty message"}

    messages.append({"role": "user", "content": user_msg})
    ai_msg = call_gpt(messages, tools=[tool_availability, tool_price])
    messages.append({"role":"assistant","content":ai_msg.content,"function_call":ai_msg.function_call})

    tool_response = try_handle_tool_call(ai_msg)
    if tool_response:
        messages.append(tool_response)
        ai_msg2 = call_gpt(messages, tools=[tool_availability, tool_price])
        messages.append({"role":"assistant","content":ai_msg2.content,"function_call":ai_msg2.function_call})
        return {"reply": ai_msg2.content}
    return {"reply": ai_msg.content}
