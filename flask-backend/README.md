# Flask Backend — SmartMine AI Safety Detection

A lightweight Flask + SQLite persistence and AI-chat layer for the SmartMine
project. It runs **independently** of the existing FastAPI inference backend
(`backend/api.py`).

## Architecture

| Service | Port | Responsibility |
|---------|------|----------------|
| FastAPI (`backend/api.py`) | 8000 | ResNet-101 image inference |
| Flask (`flask-backend/app.py`) | 5001 | DB persistence + Gemini AI chat |
| Next.js (`app/`) | 3000 | Frontend UI |

## Setup

### 1. Install dependencies

```bash
cd flask-backend
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the `flask-backend/` directory (or set the variables
in your shell):

```env
# Required for the /api/ai/chat endpoint
GEMINI_API_KEY=your_gemini_api_key_here
```

Obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Run the server

```bash
python app.py
```

The server starts on **http://localhost:5001**.  
The SQLite database (`smartmine.db`) is created automatically on first run.

---

## API Reference

### Users

| Method | Path | Body / Params | Description |
|--------|------|---------------|-------------|
| `POST` | `/api/users` | `{ name, email }` | Create (or return existing) user |
| `GET` | `/api/users/<id>` | — | Fetch user profile |

### Predictions

| Method | Path | Body / Params | Description |
|--------|------|---------------|-------------|
| `POST` | `/api/predictions` | `{ user_id, filename, prediction, confidence, all_probabilities }` | Store a prediction result |
| `GET` | `/api/predictions` | `?user_id=` (optional) | List all predictions |
| `GET` | `/api/predictions/<id>` | — | Fetch one prediction record |

### Chat History

| Method | Path | Body / Params | Description |
|--------|------|---------------|-------------|
| `POST` | `/api/chat` | `{ user_id, role, content }` | Store a chat message (`role`: `user`\|`assistant`) |
| `GET` | `/api/chat/<user_id>` | — | Fetch full chat history for a user |

### AI Chat (Gemini)

| Method | Path | Body | Description |
|--------|------|------|-------------|
| `POST` | `/api/ai/chat` | `{ user_id, message, context? }` | Send a message to Gemini; returns `{ reply, user_message_id, assistant_message_id }` |

The optional `context` field should be the prediction result object
(`{ prediction, confidence, all_probabilities }`) returned by the FastAPI
backend. It is automatically prepended to the prompt so that Gemini can give
context-aware answers.

---

## Example cURL calls

```bash
# Create a user
curl -X POST http://localhost:5001/api/users \
  -H 'Content-Type: application/json' \
  -d '{"name":"Alice","email":"alice@example.com"}'

# Store a prediction
curl -X POST http://localhost:5001/api/predictions \
  -H 'Content-Type: application/json' \
  -d '{"user_id":1,"filename":"mine01.jpg","prediction":"safe","confidence":0.92,"all_probabilities":{"safe":0.92,"unsafe":0.04,"helmet":0.03,"hazard":0.01}}'

# Chat with Gemini about the result
curl -X POST http://localhost:5001/api/ai/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":1,"message":"What does this result mean?","context":{"prediction":"safe","confidence":0.92}}'
```
