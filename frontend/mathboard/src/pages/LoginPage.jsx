import { useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";

import { useAuth } from "../contexts/AuthContext.jsx";

export default function LoginPage() {
  const { user, configured, signInWithGoogle, signInWithEmail, signUpWithEmail } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const redirectTo = location.state?.from?.pathname ?? "/notebooks";

  const [mode, setMode] = useState("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  if (user) return <Navigate to={redirectTo} replace />;

  const handleGoogle = async () => {
    setError(null);
    setSubmitting(true);
    try {
      await signInWithGoogle();
      navigate(redirectTo, { replace: true });
    } catch (err) {
      setError(prettifyAuthError(err));
    } finally {
      setSubmitting(false);
    }
  };

  const handleEmailSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      if (mode === "signup") {
        await signUpWithEmail(email, password);
      } else {
        await signInWithEmail(email, password);
      }
      navigate(redirectTo, { replace: true });
    } catch (err) {
      setError(prettifyAuthError(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="auth-page">
      {/* ── Left decorative panel ── */}
      <div className="auth-left">
        <div>
          <span className="smallcaps">↳ Welcome back</span>
          <p className="quote">
            "Pen on paper, but the{" "}
            <span className="acc">paper</span> can{" "}
            <span className="it">do the algebra.</span>"
          </p>
          <div className="who">— Mathboard, since 2024</div>
        </div>
        <div className="marg">
          <div className="c">
            <div className="l">Recognition accuracy</div>
            <div className="v">98.4%</div>
          </div>
          <div className="c">
            <div className="l">Symbol classes</div>
            <div className="v">110+</div>
          </div>
          <div className="c">
            <div className="l">Avg latency</div>
            <div className="v">312ms</div>
          </div>
        </div>
      </div>

      {/* ── Right form panel ── */}
      <div className="auth-right">
        <span className="smallcaps">№ 02 — {mode === "signup" ? "Create account" : "Sign in"}</span>
        <h2 style={{ marginTop: 8 }}>
          Open your <span className="it">boards.</span>
        </h2>
        <p className="sub">Use your email, or continue with Google.</p>

        {!configured && (
          <div className="banner banner-warn" role="alert" style={{ marginBottom: 20 }}>
            <div>
              Firebase isn&apos;t configured. Copy <code>.env.local.example</code> →{" "}
              <code>.env.local</code>, fill in your <code>VITE_FIREBASE_*</code> keys, restart.
            </div>
          </div>
        )}

        <button
          type="button"
          className="auth-google-btn"
          onClick={handleGoogle}
          disabled={!configured || submitting}
        >
          <GoogleMark />
          <span>Continue with Google</span>
        </button>

        <div className="auth-divider">or continue with email</div>

        <form onSubmit={handleEmailSubmit} className="auth-form">
          <label className="auth-field">
            <span>Email address</span>
            <input
              type="email"
              placeholder="ada@lovelace.io"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={!configured || submitting}
            />
          </label>
          <label className="auth-field">
            <span>Password</span>
            <input
              type="password"
              placeholder="••••••••••"
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
              required
              minLength={6}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={!configured || submitting}
            />
          </label>

          {error && (
            <div className="auth-error" role="alert">{error}</div>
          )}

          <button
            type="submit"
            className="convert-btn"
            style={{ marginTop: 8 }}
            disabled={!configured || submitting}
          >
            <span>
              {submitting
                ? "Working…"
                : mode === "signup"
                  ? "Create account →"
                  : "Sign in →"}
            </span>
          </button>
        </form>

        <button
          type="button"
          className="auth-toggle"
          onClick={() => { setMode(m => m === "signup" ? "signin" : "signup"); setError(null); }}
        >
          {mode === "signup"
            ? "Already have an account? Sign in"
            : "New here? Create an account"}
        </button>
      </div>
    </main>
  );
}

function GoogleMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.49h4.84a4.14 4.14 0 0 1-1.79 2.72v2.26h2.9c1.7-1.57 2.69-3.88 2.69-6.63z"/>
      <path fill="#34A853" d="M9 18c2.43 0 4.47-.81 5.96-2.18l-2.9-2.26c-.8.54-1.83.86-3.06.86-2.35 0-4.34-1.59-5.05-3.72H.95v2.33A8.999 8.999 0 0 0 9 18z"/>
      <path fill="#FBBC05" d="M3.95 10.7A5.41 5.41 0 0 1 3.66 9c0-.59.1-1.16.29-1.7V4.97H.95A8.997 8.997 0 0 0 0 9c0 1.45.35 2.82.95 4.03l3-2.33z"/>
      <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0A8.999 8.999 0 0 0 .95 4.97l3 2.33C4.66 5.17 6.65 3.58 9 3.58z"/>
    </svg>
  );
}

function prettifyAuthError(err) {
  const code = err?.code ?? "";
  if (code === "auth/invalid-credential" || code === "auth/wrong-password") return "Email or password is incorrect.";
  if (code === "auth/user-not-found")       return "No account found with that email.";
  if (code === "auth/email-already-in-use") return "An account with that email already exists.";
  if (code === "auth/weak-password")        return "Password must be at least 6 characters.";
  if (code === "auth/popup-closed-by-user") return "Sign-in was cancelled.";
  return err?.message ?? "Something went wrong. Try again.";
}
