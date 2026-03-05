"""Flask backend for SmartMine AI Safety Detection.

Provides database persistence (SQLite via SQLAlchemy) and Gemini-powered
AI chat for the SmartMine Next.js frontend.

Run with:
    python app.py
or:
    flask --app app run --port 5001
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

app = Flask(__name__)

# CORS — allow all origins for local development
CORS(app)

# SQLite database stored alongside this file
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{os.path.join(_BASE_DIR, 'smartmine.db')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class User(db.Model):  # type: ignore[misc]
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    email = db.Column(db.String(256), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    predictions = db.relationship("PredictionRecord", backref="user", lazy=True)
    chat_messages = db.relationship("ChatMessage", backref="user", lazy=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PredictionRecord(db.Model):  # type: ignore[misc]
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    filename = db.Column(db.String(256), nullable=False)
    prediction = db.Column(db.String(64), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    all_probabilities = db.Column(db.Text, nullable=False)  # JSON-encoded
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "filename": self.filename,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "all_probabilities": json.loads(self.all_probabilities),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ChatMessage(db.Model):  # type: ignore[misc]
    __tablename__ = "chat_messages"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    role = db.Column(db.String(16), nullable=False)  # "user" | "assistant"
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# DB initialisation — create tables on first import
# ---------------------------------------------------------------------------

with app.app_context():
    db.create_all()

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------

_GEMINI_SYSTEM_PROMPT = (
    "You are a mine safety AI assistant. Help users understand their safety "
    "detection results. Classes are: safe (worker/area is safe), unsafe "
    "(potential danger), helmet (PPE detected), hazard (environmental hazard "
    "detected). Be concise, helpful, and safety-focused."
)


def _get_gemini_model():
    """Return a configured Gemini GenerativeModel or None if key missing."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=_GEMINI_SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------------
# Routes — Users
# ---------------------------------------------------------------------------


@app.route("/api/users", methods=["POST"])
def create_user():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()

    if not name or not email:
        return jsonify({"error": "name and email are required"}), 400

    existing = User.query.filter_by(email=email).first()
    if existing:
        return jsonify(existing.to_dict()), 200

    user = User(name=name, email=email)
    db.session.add(user)
    db.session.commit()
    return jsonify(user.to_dict()), 201


@app.route("/api/users/<int:user_id>", methods=["GET"])
def get_user(user_id: int):
    user = db.session.get(User, user_id)
    if user is None:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.to_dict()), 200


# ---------------------------------------------------------------------------
# Routes — Predictions
# ---------------------------------------------------------------------------


@app.route("/api/predictions", methods=["POST"])
def create_prediction():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    filename = (data.get("filename") or "").strip()
    prediction = (data.get("prediction") or "").strip()
    confidence = data.get("confidence")
    all_probabilities = data.get("all_probabilities")

    if not all([user_id, filename, prediction, confidence is not None, all_probabilities]):
        return jsonify({"error": "user_id, filename, prediction, confidence, and all_probabilities are required"}), 400

    if db.session.get(User, user_id) is None:
        return jsonify({"error": "User not found"}), 404

    record = PredictionRecord(
        user_id=user_id,
        filename=filename,
        prediction=prediction,
        confidence=float(confidence),
        all_probabilities=json.dumps(all_probabilities),
    )
    db.session.add(record)
    db.session.commit()
    return jsonify(record.to_dict()), 201


@app.route("/api/predictions", methods=["GET"])
def list_predictions():
    user_id = request.args.get("user_id", type=int)
    query = PredictionRecord.query
    if user_id is not None:
        query = query.filter_by(user_id=user_id)
    records = query.order_by(PredictionRecord.created_at.desc()).all()
    return jsonify([r.to_dict() for r in records]), 200


@app.route("/api/predictions/<int:prediction_id>", methods=["GET"])
def get_prediction(prediction_id: int):
    record = db.session.get(PredictionRecord, prediction_id)
    if record is None:
        return jsonify({"error": "Prediction record not found"}), 404
    return jsonify(record.to_dict()), 200


# ---------------------------------------------------------------------------
# Routes — Chat History
# ---------------------------------------------------------------------------


@app.route("/api/chat", methods=["POST"])
def store_chat_message():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    role = (data.get("role") or "").strip()
    content = (data.get("content") or "").strip()

    if not all([user_id, role, content]):
        return jsonify({"error": "user_id, role, and content are required"}), 400

    if role not in ("user", "assistant"):
        return jsonify({"error": "role must be 'user' or 'assistant'"}), 400

    if db.session.get(User, user_id) is None:
        return jsonify({"error": "User not found"}), 404

    msg = ChatMessage(user_id=user_id, role=role, content=content)
    db.session.add(msg)
    db.session.commit()
    return jsonify(msg.to_dict()), 201


@app.route("/api/chat/<int:user_id>", methods=["GET"])
def get_chat_history(user_id: int):
    if db.session.get(User, user_id) is None:
        return jsonify({"error": "User not found"}), 404
    messages = (
        ChatMessage.query.filter_by(user_id=user_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return jsonify([m.to_dict() for m in messages]), 200


# ---------------------------------------------------------------------------
# Routes — AI Chat (Gemini)
# ---------------------------------------------------------------------------


@app.route("/api/ai/chat", methods=["POST"])
def ai_chat():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    message = (data.get("message") or "").strip()
    context = data.get("context")  # optional prediction result object

    if not user_id or not message:
        return jsonify({"error": "user_id and message are required"}), 400

    if db.session.get(User, user_id) is None:
        return jsonify({"error": "User not found"}), 404

    model = _get_gemini_model()
    if model is None:
        return jsonify({"error": "Gemini API key not configured"}), 503

    # Build prompt, optionally including prediction context
    prompt = message
    if context:
        context_text = (
            f"Prediction context: prediction={context.get('prediction')}, "
            f"confidence={context.get('confidence')}, "
            f"all_probabilities={context.get('all_probabilities')}. "
        )
        prompt = context_text + message

    try:
        response = model.generate_content(prompt)
        reply = response.text
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Gemini API error: {exc}"}), 502

    # Persist both messages
    user_msg = ChatMessage(user_id=user_id, role="user", content=message)
    assistant_msg = ChatMessage(user_id=user_id, role="assistant", content=reply)
    db.session.add(user_msg)
    db.session.add(assistant_msg)
    db.session.commit()

    return jsonify(
        {
            "reply": reply,
            "user_message_id": user_msg.id,
            "assistant_message_id": assistant_msg.id,
        }
    ), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
