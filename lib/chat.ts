/**
 * SmartMine AI Safety Detection — Flask backend client module.
 *
 * Provides typed helpers for communicating with the Flask persistence /
 * AI-chat backend (`flask-backend/app.py`).
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ChatMessage {
  id?: number;
  role: "user" | "assistant";
  content: string;
  created_at?: string;
}

export interface AIChatResponse {
  reply: string;
  user_message_id: number;
  assistant_message_id: number;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const FLASK_BASE_URL =
  process.env.NEXT_PUBLIC_FLASK_API_URL ?? "http://localhost:5001";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function flaskFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${FLASK_BASE_URL}${path}`;
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body?.error) detail = body.error;
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(`Flask API error (${response.status}): ${detail}`);
  }

  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Users
// ---------------------------------------------------------------------------

/**
 * Create a new user (or return existing one if the e-mail is already taken).
 */
export async function createUser(
  name: string,
  email: string,
): Promise<{ id: number; name: string; email: string }> {
  return flaskFetch("/api/users", {
    method: "POST",
    body: JSON.stringify({ name, email }),
  });
}

// ---------------------------------------------------------------------------
// Predictions
// ---------------------------------------------------------------------------

/**
 * Persist a prediction result in the Flask database.
 */
export async function savePrediction(
  userId: number,
  filename: string,
  prediction: string,
  confidence: number,
  allProbabilities: Record<string, number>,
): Promise<{ id: number }> {
  return flaskFetch("/api/predictions", {
    method: "POST",
    body: JSON.stringify({
      user_id: userId,
      filename,
      prediction,
      confidence,
      all_probabilities: allProbabilities,
    }),
  });
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

/**
 * Send a message to the Gemini AI chat endpoint and persist both turns.
 *
 * @param userId  - ID of the current user (from Flask `/api/users`).
 * @param message - The user's question / message text.
 * @param context - Optional prediction result to include as context.
 */
export async function sendChatMessage(
  userId: number,
  message: string,
  context?: object,
): Promise<AIChatResponse> {
  return flaskFetch("/api/ai/chat", {
    method: "POST",
    body: JSON.stringify({ user_id: userId, message, context }),
  });
}

/**
 * Retrieve the full chat history for a user from the Flask database.
 */
export async function getChatHistory(userId: number): Promise<ChatMessage[]> {
  return flaskFetch<ChatMessage[]>(`/api/chat/${userId}`);
}
