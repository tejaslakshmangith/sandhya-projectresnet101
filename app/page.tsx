"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  checkHealth,
  predictImage,
  type HealthStatus,
  type PredictionResult,
} from "@/lib/ai";
import {
  createUser,
  savePrediction,
  sendChatMessage,
  type ChatMessage,
} from "@/lib/chat";

// ---------------------------------------------------------------------------
// Class colour map
// ---------------------------------------------------------------------------
const CLASS_COLORS: Record<string, string> = {
  safe: "bg-emerald-500",
  unsafe: "bg-red-500",
  helmet: "bg-amber-400",
  hazard: "bg-orange-500",
};

const CLASS_BADGES: Record<string, string> = {
  safe: "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40",
  unsafe: "bg-red-500/20 text-red-300 border border-red-500/40",
  helmet: "bg-amber-400/20 text-amber-300 border border-amber-400/40",
  hazard: "bg-orange-500/20 text-orange-300 border border-orange-500/40",
};

function colorFor(cls: string): string {
  return CLASS_COLORS[cls] ?? "bg-blue-500";
}

function badgeFor(cls: string): string {
  return CLASS_BADGES[cls] ?? "bg-blue-500/20 text-blue-300 border border-blue-500/40";
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusBadge({ health }: { health: HealthStatus | null }) {
  if (!health) {
    return (
      <span className="flex items-center gap-1.5 text-xs text-gray-500">
        <span className="size-2 rounded-full bg-gray-600 animate-pulse" />
        Checking backend…
      </span>
    );
  }
  if (health.model_loaded) {
    return (
      <span className="flex items-center gap-1.5 text-xs text-emerald-400">
        <span className="size-2 rounded-full bg-emerald-400" />
        Model ready
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1.5 text-xs text-amber-400">
      <span className="size-2 rounded-full bg-amber-400" />
      Backend up — model not loaded
    </span>
  );
}

function ConfidenceBar({
  label,
  value,
  isTop,
}: {
  label: string;
  value: number;
  isTop: boolean;
}) {
  const pct = (value * 100).toFixed(1);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className={`font-medium ${isTop ? "text-white" : "text-gray-400"}`}>
          {label}
        </span>
        <span className={isTop ? "text-white font-semibold" : "text-gray-500"}>
          {pct}%
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-gray-800">
        <div
          className={`h-2 rounded-full transition-all duration-700 ${colorFor(label)}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// AI Chat Panel sub-component
// ---------------------------------------------------------------------------

const INITIAL_ASSISTANT_MSG: ChatMessage = {
  role: "assistant",
  content:
    "Hi! I can answer questions about your mine-safety detection result. What would you like to know?",
};

function AIChatPanel({
  userId,
  predictionContext,
}: {
  userId: number | null;
  predictionContext: PredictionResult | null;
}) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([INITIAL_ASSISTANT_MSG]);
  const [input, setInput] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the latest message
  useEffect(() => {
    if (open) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, open]);

  // Reset panel when prediction context changes
  useEffect(() => {
    setMessages([INITIAL_ASSISTANT_MSG]);
    setChatError(null);
    setInput("");
  }, [predictionContext]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || aiLoading) return;

    const userMsg: ChatMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setAiLoading(true);
    setChatError(null);

    try {
      if (userId !== null) {
        const response = await sendChatMessage(
          userId,
          text,
          predictionContext ?? undefined,
        );
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: response.reply },
        ]);
      } else {
        // Flask backend not available — show a graceful fallback
        throw new Error("User session not initialised. Is the Flask backend running?");
      }
    } catch (err: unknown) {
      setChatError(
        err instanceof Error ? err.message : "Failed to get AI response.",
      );
      setMessages((prev) => prev.slice(0, -1)); // remove optimistic user msg
      setInput(text);
    } finally {
      setAiLoading(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="rounded-2xl bg-gray-900 border border-gray-800 overflow-hidden">
      {/* Toggle header */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-gray-800 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-lg">💬</span>
          <span className="font-semibold text-white">Ask AI about this result</span>
        </div>
        <span className="text-gray-400 text-sm">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="border-t border-gray-800">
          {/* Message history */}
          <div className="h-72 overflow-y-auto px-4 py-4 space-y-3 bg-gray-950">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-2 text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "bg-blue-600 text-white rounded-tr-sm"
                      : "bg-gray-800 text-gray-200 rounded-tl-sm"
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            {aiLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-800 text-gray-400 rounded-2xl rounded-tl-sm px-4 py-2 text-sm italic">
                  AI is thinking…
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Error */}
          {chatError && (
            <div className="mx-4 my-2 rounded-lg bg-red-500/10 border border-red-500/30 px-3 py-2 text-red-300 text-xs">
              <strong>Error: </strong>{chatError}
            </div>
          )}

          {/* Input */}
          <div className="px-4 py-3 border-t border-gray-800 flex gap-2 bg-gray-900">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask a question about the result…"
              disabled={aiLoading}
              className="flex-1 rounded-xl bg-gray-800 border border-gray-700 px-4 py-2 text-sm text-white
                placeholder:text-gray-500 focus:outline-none focus:border-blue-500 disabled:opacity-50"
            />
            <button
              onClick={sendMessage}
              disabled={aiLoading || !input.trim()}
              className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white text-sm
                font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function SmartMinePage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Guest user session (persisted in localStorage)
  const [userId, setUserId] = useState<number | null>(null);

  // Initialise guest user on first render (client-side only)
  useEffect(() => {
    const stored = localStorage.getItem("smartmine_user_id");
    const parsedId = stored ? parseInt(stored, 10) : NaN;
    if (!isNaN(parsedId) && parsedId > 0) {
      setUserId(parsedId);
      return;
    }
    const ts = Date.now();
    createUser("Guest", `guest_${ts}@smartmine.local`)
      .then((u) => {
        localStorage.setItem("smartmine_user_id", String(u.id));
        setUserId(u.id);
      })
      .catch(() => {
        // Flask backend not available — continue without persistence
      });
  }, []);

  // Poll backend health on mount
  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() =>
        setHealth({ status: "unreachable", model_loaded: false })
      );
  }, []);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file (JPEG, PNG, WebP, …).");
      return;
    }
    setSelectedFile(file);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(file);
    setPreview(url);
  }, []);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const onAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictImage(selectedFile);
      setResult(res);

      // Persist prediction to Flask backend (best-effort)
      if (userId) {
        savePrediction(
          userId,
          selectedFile.name,
          res.prediction,
          res.confidence,
          res.all_probabilities,
        ).catch(() => {
          // Ignore persistence errors — they don't affect the UI
        });
      }
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred. Make sure the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const onReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  // Sort probabilities highest-first
  const sortedProbs = result
    ? Object.entries(result.all_probabilities).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <main className="mx-auto max-w-3xl px-4 py-12 space-y-8">
      {/* Header */}
      <header className="text-center space-y-2">
        <div className="flex items-center justify-center gap-3 mb-4">
          <div className="size-10 rounded-xl bg-amber-500 flex items-center justify-center text-xl font-bold text-gray-900">
            ⛏
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-white">
            SmartMine AI Safety Detection
          </h1>
        </div>
        <p className="text-gray-400 max-w-lg mx-auto">
          Upload a mine-site image and our ResNet-101 model will classify it as{" "}
          <span className="text-emerald-400 font-medium">safe</span>,{" "}
          <span className="text-red-400 font-medium">unsafe</span>,{" "}
          <span className="text-amber-300 font-medium">helmet</span>, or{" "}
          <span className="text-orange-400 font-medium">hazard</span>.
        </p>
        <div className="flex justify-center mt-2">
          <StatusBadge health={health} />
        </div>
      </header>

      {/* Upload zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !preview && inputRef.current?.click()}
        className={`relative rounded-2xl border-2 border-dashed transition-colors cursor-pointer select-none
          ${dragging ? "border-amber-500 bg-amber-500/10" : "border-gray-700 bg-gray-900 hover:border-gray-500"}
          ${preview ? "cursor-default" : ""}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="sr-only"
          onChange={onInputChange}
        />

        {preview ? (
          <div className="relative">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={preview}
              alt="Uploaded mine-site image"
              className="w-full max-h-80 object-contain rounded-2xl"
            />
            <button
              onClick={(e) => { e.stopPropagation(); onReset(); }}
              className="absolute top-3 right-3 size-8 rounded-full bg-gray-900/80 flex items-center justify-center text-gray-300 hover:text-white hover:bg-gray-800 transition-colors"
              aria-label="Remove image"
            >
              ✕
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-16 px-8 gap-3 text-center">
            <div className="text-4xl">🖼️</div>
            <p className="text-gray-300 font-medium">
              Drag &amp; drop an image here
            </p>
            <p className="text-gray-500 text-sm">or click to browse your files</p>
            <p className="text-gray-600 text-xs mt-1">
              Supports JPEG, PNG, WebP, BMP, GIF
            </p>
          </div>
        )}
      </div>

      {/* Analyze button */}
      {selectedFile && !result && (
        <button
          onClick={onAnalyze}
          disabled={loading}
          className="w-full py-3 rounded-xl font-semibold text-gray-900 bg-amber-500 hover:bg-amber-400 active:bg-amber-600
            disabled:opacity-60 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <span className="size-4 border-2 border-gray-900/40 border-t-gray-900 rounded-full animate-spin" />
              Analyzing…
            </>
          ) : (
            "Analyze Image"
          )}
        </button>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-xl bg-red-500/10 border border-red-500/30 px-4 py-3 text-red-300 text-sm">
          <strong className="font-semibold">Error: </strong>{error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="rounded-2xl bg-gray-900 border border-gray-800 overflow-hidden">
          {/* Top result */}
          <div className="px-6 py-5 border-b border-gray-800 flex items-center justify-between gap-4">
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">
                Prediction
              </p>
              <div className="flex items-center gap-3">
                <span
                  className={`px-3 py-1 rounded-lg text-sm font-bold ${badgeFor(result.prediction)}`}
                >
                  {result.prediction.toUpperCase()}
                </span>
                <span className="text-2xl font-bold text-white">
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            <button
              onClick={onReset}
              className="text-sm text-gray-500 hover:text-gray-300 transition-colors underline underline-offset-2"
            >
              Try another image
            </button>
          </div>

          {/* All probabilities */}
          <div className="px-6 py-5 space-y-3">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">
              All class probabilities
            </p>
            {sortedProbs.map(([cls, prob]) => (
              <ConfidenceBar
                key={cls}
                label={cls}
                value={prob}
                isTop={cls === result.prediction}
              />
            ))}
          </div>
        </div>
      )}

      {/* AI Chat Panel — shown after a prediction */}
      {result && (
        <AIChatPanel userId={userId} predictionContext={result} />
      )}

      {/* Footer */}
      <footer className="text-center text-xs text-gray-600 pt-4">
        Powered by ResNet-101 · FastAPI · Next.js
      </footer>
    </main>
  );
}
