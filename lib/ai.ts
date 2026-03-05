/**
 * SmartMine AI Safety Detection — Next.js TypeScript client module.
 *
 * Provides typed helpers for communicating with the FastAPI backend
 * (`backend/api.py`).
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PredictionResult {
  prediction: string;
  confidence: number;
  all_probabilities: Record<string, number>;
  status: string;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
}

export interface ClassesResponse {
  classes: string[];
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const API_BASE_URL =
  process.env.NEXT_PUBLIC_AI_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const response = await fetch(url, init);

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(`SmartMine API error (${response.status}): ${detail}`);
  }

  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Classify a mine-safety image.
 *
 * @param imageFile - A `File` object (e.g. from an `<input type="file">` element).
 * @returns Prediction result with class name, confidence, and per-class probabilities.
 */
export async function predictImage(imageFile: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", imageFile);

  return apiFetch<PredictionResult>("/predict", {
    method: "POST",
    body: formData,
  });
}

/**
 * Check whether the FastAPI backend is healthy and the model is loaded.
 */
export async function checkHealth(): Promise<HealthStatus> {
  return apiFetch<HealthStatus>("/health");
}

/**
 * Retrieve the list of class names the model was trained on.
 */
export async function getClasses(): Promise<string[]> {
  const response = await apiFetch<ClassesResponse>("/classes");
  return response.classes;
}
