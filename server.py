# Import for API serving
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import csv
from datetime import datetime
import uvicorn

# Import for LLM
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing_extensions import TypedDict
from typing import Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolExecutor, ToolNode, tools_condition


# Attempt to get API key from env variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    # If you don't have OPENAI_API_KEY set in env variable, please put your API key here to try
    OPENAI_API_KEY = ""

CSV_FILE = './data/appointments.csv'

MAIN_AGENT_PROMPT = """### You are an expert appointment booking agent for a spa establishment.

You will receive messages from customers asking to book an appointment. At any given time, there can only be one appointment. 

Your tasks include:
1. Providing available time slots when customers ask for availability during a certain time period.
2. Offering alternative time slots if the requested time is already booked.
3. Ask for the following data at the end of your first response: 
- Get customer's name
- Get appointment's duration (30 minutes, 45 minutes, 60 minutes, or 90 minutes duration)
- Get appointment start time
4. Booking the appointment if the customer decides to proceed. Please make sure to follow the booking policies!
5. When you decide to book appointment, please do things in this following order:
- Make sure you have the customer's name
- Make sure you have the appointment's duration (30 minutes, 45 minutes, 60 minutes, or 90 minutes duration)
- Make sure you have the appointment's start time
- Call book_appointment function
6. You are not capable to view, reschedule, or cancel existing booking. Your job is only to help customer book a NEW appointment.


### Booking policies
- Business hours starts at 9 AM and ends at 6 PM (Bali's time/GMT+8)
- Customers can book between 30 minutes, 45 minutes, 60 minutes, and 90 minutes duration
- Start time for booking can only be done in minute 00,15,30,45. Example of valid start time options: 10.00, 10.15. Example of INVALID options: 10.03, 10.37
- Bookings for past dates and times are not allowed.
   
### Instructions:
- Be polite and professional in your interactions.
- Always confirm the details of the booking with the customer.
- Ensure that there are no overlapping appointments before booking a new one.
- Offer alternative slots that are close to the requested time if the preferred slot is unavailable.
- Confirm the booking with a summary of the details once the appointment is successfully booked.

### Today's date
{date_today}

### Current Time Information
{current_time}
"""


##########################
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
###########################
# Define Pydantic models for the request and response bodies
class InnerMessage(BaseModel):
    content: str
    id: str

class Message(BaseModel):
    message: InnerMessage

class ResponseMessage(BaseModel):
    message: InnerMessage

###########################
# LLM Chatbot
###########################
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Tools for ChatBot to Call
@tool
def get_available_time(
    date_to_check: Annotated[str, "date to be checked for its time availability"]
) -> str:
    """Call to get available time slot during date_to_check"""

    # Define opening hours
    opening_hour = datetime.strptime(f"{date_to_check} 09:00:00", "%Y-%m-%d %H:%M:%S")
    closing_hour = datetime.strptime(f"{date_to_check} 18:00:00", "%Y-%m-%d %H:%M:%S")
    
    # Get current time in GMT+8 and convert to naive datetime
    now = datetime.now(timezone.utc) + timedelta(hours=8)
    now = now.replace(tzinfo=None)

    date_to_check_dt = datetime.strptime(date_to_check, "%Y-%m-%d").date()

    # Handle date checks
    print(now)
    if date_to_check_dt < now.date():
        return f"{date_to_check} is in the past. Bookings for past dates are not allowed."
    
    # If date_to_check is today, adjust opening and closing hours (prevent booking on past time)
    
    if now.date() == date_to_check_dt: 
        opening_hour = max(opening_hour, now)
        if now > closing_hour:
            return "The current time is past closing hours. No available slots today."
    
    # Read CSV file
    df = pd.read_csv(CSV_FILE)
    
    # Convert 'Date', 'Start', and 'End' columns to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    df['Start'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Start'])
    df['End'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['End'])
    
    # Filter the DataFrame for the given date_to_check
    df = df[df['Date'].dt.date == datetime.strptime(date_to_check, "%Y-%m-%d").date()]
    
    # Initialize available time slots
    available_slots = []
    
    # Find gaps between booked slots
    current_time = opening_hour
    for _, row in df.sort_values(by='Start').iterrows():
        start_time = row['Start']
        if current_time < start_time:
            available_slots.append((current_time, start_time))
        current_time = max(current_time, row['End'])
    
    # Check for slot between last booking and closing hour
    if current_time < closing_hour:
        available_slots.append((current_time, closing_hour))
    if (available_slots)==[]:
        return f"No timeslot is available for {date_to_check}. Offer customer to book for another date!"
    
    # Format available slots
    available_slots_str = ', '.join([f"{slot[0].strftime('%H.%M')}-{slot[1].strftime('%H.%M')}" for slot in available_slots])
    
    return f"{date_to_check} available at: {available_slots_str}"

@tool
def book_appointment(
    name: Annotated[str, "the name of the customer"], 
    date: Annotated[str, "date of appointment"], 
    start_time: Annotated[str, "start time of appointment, in %H:%M format"], 
    end_time: Annotated[str, "end time of appointment, in %H:%M format"]
) -> str:
    """Function to check if appointment slot is available and book the appointment"""

    # Load current appointments
    with open(CSV_FILE, mode='r') as file:
        appointments = list(csv.DictReader(file))

    # Ensure the datetime strings are correctly formatted
    requested_start = datetime.strptime(f"{date} {start_time}:00", "%Y-%m-%d %H:%M:%S")
    requested_end = datetime.strptime(f"{date} {end_time}:00", "%Y-%m-%d %H:%M:%S")

    for appointment in appointments:
        appointment_start = datetime.strptime(f"{appointment['Date']} {appointment['Start']}", "%Y-%m-%d %H:%M:%S")
        appointment_end = datetime.strptime(f"{appointment['Date']} {appointment['End']}", "%Y-%m-%d %H:%M:%S")

        # Check for overlap
        if requested_start < appointment_end and requested_end > appointment_start:
            return "Slot already booked, please choose another one."

    # Book the appointment
    new_appointment = {
        "Name": name,
        "Date": date,
        "Start": f"{start_time}:00",
        "End": f"{end_time}:00"
    }
    appointments.append(new_appointment)

    # Write appointment to CSV file if booking is valid
    with open(CSV_FILE, mode='w', newline='') as file:
        fieldnames = ['Name', 'Date', 'Start', 'End']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(appointments)
    
    return f"Booking for {name} on {date} from {start_time} to {end_time} is successful!"

# Chatbot Definition
def chatbot(state: State):
    today = datetime.now(timezone(timedelta(hours=8))).strftime("Today is %A, %d %B %Y")
    time_now = datetime.now(timezone(timedelta(hours=8))).strftime("The current time is %H:%M:%S GMT+8 (Bali Time)")

    user_message = HumanMessage(content=f"{state['messages']}")
    messages = [
        SystemMessage(
            content=MAIN_AGENT_PROMPT.format(date_today=today, current_time=time_now)
        ),
        user_message
    ]
    response = llm.invoke(messages)
    return {"messages": [response]}

tools = [get_available_time, book_appointment]
tool_executor = ToolExecutor(tools)
graph_builder = StateGraph(State)
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0, openai_api_key=OPENAI_API_KEY)
llm = llm.bind_tools(tools)

# Define Node
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

# Define edges
graph_builder.set_entry_point("chatbot")
graph_builder.add_conditional_edges(
    "chatbot", tools_condition
)
graph_builder.add_edge("tools", "chatbot")


memory = SqliteSaver.from_conn_string(":memory:")
graph = graph_builder.compile(checkpointer=memory)

def chat_once(user_input, session_id):
    config = {"configurable": {"thread_id": session_id}}
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        event["messages"][-1].pretty_print()
    return event["messages"][-1]


###########################
# Expose /chat endpoint
###########################
@app.post("/chat", response_model=ResponseMessage)
async def chat_endpoint(request_body: Message):
    # Access the data from the request body
    message_content = request_body.message.content
    chat_id = request_body.message.id
    
    # Assuming chat_once is a function that returns a string
    response_text = chat_once(message_content, chat_id)
    # Create response message with `message` key
    response = ResponseMessage(message=InnerMessage(content=response_text.content, id=chat_id))
    return response


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
