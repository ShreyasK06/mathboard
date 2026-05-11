import { useEffect, useState } from "react";

import { BACKEND_URL } from "../services/backend.js";

// Module-level cache so the health check fires once per page load,
// not on every component mount or navigation.
let _cache = null;
let _promise = null;

export function useBackendHealth() {
  const [status, setStatus] = useState(_cache?.status ?? "checking");
  const [geminiKeySet, setGeminiKeySet] = useState(_cache?.geminiKeySet ?? true);

  useEffect(() => {
    if (_cache) {
      setStatus(_cache.status);
      setGeminiKeySet(_cache.geminiKeySet);
      return;
    }
    let cancelled = false;
    if (!_promise) {
      _promise = fetch(`${BACKEND_URL}/health`, { cache: "no-store" })
        .then((res) => {
          if (!res.ok) throw new Error(`status ${res.status}`);
          return res.json();
        });
    }
    _promise.then(
      (data) => {
        _cache = { status: "up", geminiKeySet: Boolean(data.gemini_key_set) };
        if (!cancelled) {
          setStatus("up");
          setGeminiKeySet(Boolean(data.gemini_key_set));
        }
      },
      () => {
        _cache = { status: "down", geminiKeySet: false };
        if (!cancelled) setStatus("down");
      },
    );
    return () => { cancelled = true; };
  }, []);

  return { status, geminiKeySet };
}
