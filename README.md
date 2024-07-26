# Appointment Booking Chatbot
Appointment booking agent using an LLM-based chatbot powered by OpenAI GPT-4o-mini and LangGraph.  a

## Installing requirements
1. Clone this repository.
2. Navigate to the repository directory (e.g., `cd ./appointment-booking-agent`).
### Option 1: Direct Installation
1. Install Python 3.12.4 on your system.
2. Run `pip install -r requirements.txt`.

### Option 2 (Recommended): Using Conda Environment
1. Install Conda or Miniconda.
2. Run `conda env create -f environment.yml`.
3. Activate the environment with `conda activate booking_agent`.

## Running the Project
1. Start the server by running `python server.py`.
2. If the server is running, you should see the following messages:
    ```
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
    INFO:     127.0.0.1:59423 - "OPTIONS /chat HTTP/1.1" 200 OK
    ```
3. Open `index.html` in your web browser to access the UI.
