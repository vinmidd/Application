You are an expert Python developer working on a PoC project called **APES (AI Powered Enterprise Services)**. Your goal is to implement a LangGraph-based conversational system that allows users to ask ID card–related questions in natural language.

Please generate all the necessary boilerplate and logic for the following:

---

📁 **Project Structure**
Create a modular project with these folders:

- `agents/`: contains `intent_resolver.py` and `response_agent.py`
- `tools/`: contains `track_id_card.py`, `estimate_id_card.py`, and `request_id_card.py`
- `orchestrator/`: contains `router.py` and `langgraph_flow.py`
- `memory/`: contains `memory_manager.py`
- `state/`: contains `graph_state.py`
- `mock_api/`: contains `mock_api_server.py` using FastAPI and sample data
- `config/`: contains `prompt_templates.py` and `settings.py`
- `test/`: contains unit tests for each module
- `ui/`: contains a **Flask-based chatbot UI**
- `main.py`: to invoke the LangGraph end-to-end flow

---

🧠 **Functional Flow**
User enters a query like:
> "I haven't received my ID card in 5 days, where is it?"

Your system should:
1. Use an LLM to **classify intent** (e.g., `track_id_card`, `estimate_id_card`, `request_id_card`)
2. Use memory to **retain context** (multi-turn)
3. Use an **orchestration layer** to select the correct tool
4. Call the correct **mock API tool** (track/estimate/request)
5. Use a **response agent** to generate human-readable replies

---

🖥️ **UI Requirements (Flask)**
- A simple Flask app (`ui/app.py`) that serves a web form
- Page should have:
  - An input box for user messages
  - A scrollable area that shows **conversation history** (previous user + bot messages)
- History should be maintained **in session or simple in-memory structure** for PoC
- On submission, the message should:
  - Trigger the LangGraph pipeline
  - Display the updated conversation in the same page
- Bootstrap / minimal styling is fine

---

📦 **LLM Calls Required**
- `intent_resolver.py`: Prompt to classify intent with confidence
- `response_agent.py`: Prompt to format API response for end-user
- Use `PydanticOutputParser` for both to parse outputs
- Optional: `entity_extraction.py` if you want structured slot filling

---

📐 **Schemas**
Please use Pydantic for all of the following models (put them in `schemas.py`):

- `GraphState`: shared LangGraph state
- `IntentResult`
- `EntityExtractionResult`
- `TrackCardInput`, `TrackCardResponse`
- `EstimateCardInput`, `EstimateCardResponse`
- `RequestCardInput`, `RequestCardResponse`
- `FinalBotResponse`

---

🔧 **Mock API**
Use FastAPI in `mock_api/mock_api_server.py` to simulate backend:
- `GET /idcard/status`
- `GET /plan/status`
- `POST /idcard/request`

Return mocked JSON values.

---

🧪 **Tests**
For each component, generate:
- Unit test for intent resolver
- Unit test for each tool
- End-to-end test simulating a full LangGraph run

---

📋 **Prompt Templates**
Create prompt templates in `config/prompt_templates.py` for:
- Intent classification
- Response generation
- Clarification/fallback

---

📚 **Tech Stack**
- Python 3.10+
- LangChain
- LangGraph
- Pydantic
- FastAPI
- **Flask (UI only)**
- OpenAI (gpt-4 or gpt-4o)

---

📌 Notes
- This is a **PoC only**. Do not implement production-grade error handling, logging, or authentication.
- Focus on **modular, clean code**, and **developer-friendly structure** for handoff.