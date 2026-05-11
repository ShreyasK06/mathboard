import { AlertCircleIcon, AlertTriangleIcon } from "./Icons.jsx";
import { BACKEND_URL } from "../services/backend.js";

export default function BackendStatusBanner({ status, geminiKeySet }) {
  if (status === "down") {
    return (
      <div className="banner banner-error" role="alert">
        <AlertCircleIcon />
        <div>
          <strong>Backend offline.</strong> Could not reach {BACKEND_URL}. Start it with{" "}
          <code>cd backend &amp;&amp; .venv/Scripts/python -m uvicorn main:app --reload</code>.
        </div>
      </div>
    );
  }
  if (status === "up" && !geminiKeySet) {
    return (
      <div className="banner banner-warn" role="alert">
        <AlertTriangleIcon />
        <div>
          <strong>Gemini API key missing.</strong> Add{" "}
          <code>GEMINI_API_KEY=...</code> to <code>backend/.env</code> and restart the backend.
        </div>
      </div>
    );
  }
  return null;
}
