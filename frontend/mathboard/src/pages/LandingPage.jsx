import { Link, Navigate } from "react-router-dom";

import Footer from "../components/Footer.jsx";
import { useAuth } from "../contexts/AuthContext.jsx";

function HandwrittenEquation() {
  const stroke = "var(--ink)";
  return (
    <svg viewBox="0 0 540 200" width="100%" height={180} aria-label="2x squared plus 5x minus 12 equals 0">
      <g stroke={stroke} strokeWidth="3.2" fill="none" strokeLinecap="round" strokeLinejoin="round">
        <path d="M30 78 C 36 60, 78 60, 78 84 C 78 104, 32 124, 28 138 L 78 138" />
        <path d="M100 92 L 142 138" /><path d="M100 138 L 142 92" />
        <path d="M152 78 C 156 70, 178 70, 178 82 C 178 92, 152 100, 152 108 L 178 108" strokeWidth="2.4"/>
        <path d="M204 116 L 240 116" /><path d="M222 98 L 222 134" />
        <path d="M268 76 L 300 76 L 290 102 C 320 92, 322 144, 286 140" />
        <path d="M330 92 L 372 138" /><path d="M330 138 L 372 92" />
        <path d="M392 116 L 426 116" />
        <path d="M444 84 L 454 78 L 454 140" />
        <path d="M468 90 C 472 74, 504 74, 504 92 C 504 108, 466 124, 466 140 L 504 140" />
        <path d="M30 168 L 80 168" strokeWidth="2.2"/><path d="M30 180 L 80 180" strokeWidth="2.2"/>
        <ellipse cx="120" cy="174" rx="18" ry="14" />
      </g>
      <text x="430" y="62" fontFamily="Instrument Serif, serif" fontStyle="italic" fontSize="15" fill="rgba(20,19,15,.42)">solve for x</text>
      <path d="M428 64 C 410 70, 396 74, 376 80" stroke="rgba(20,19,15,.38)" strokeWidth="1" fill="none"/>
      <path d="M376 80 l 6 -4 m -6 4 l 4 6" stroke="rgba(20,19,15,.38)" strokeWidth="1" fill="none"/>
    </svg>
  );
}

export default function LandingPage() {
  const { user, configured } = useAuth();

  if (user) return <Navigate to="/notebooks" replace />;

  return (
    <main className="landing">
      {/* ── HERO ──────────────────────────────────────────────── */}
      <section className="hero-section">
        <div className="section-tag">
          <span className="num">01 / OVERVIEW</span>
          <span className="line" />
          <span className="num">v 2.4 · Public beta</span>
        </div>

        <div className="hero-grid">
          <div>
            <h1>
              A whiteboard<br />
              that <span className="it">thinks</span><br />
              <span className="strike">like a</span> <span className="it">in math.</span>
            </h1>
            <p className="lede">
              Draw an equation. Press solve. Mathboard reads your handwriting,
              parses the LaTeX, and works out every step — on a canvas that saves itself.
            </p>
            <div className="cta-row">
              {configured ? (
                <Link to="/login" className="btn accent">
                  Start a free board <span className="arrow">→</span>
                </Link>
              ) : (
                <div className="banner banner-warn" role="alert" style={{ marginBottom: 0 }}>
                  <div>
                    Firebase isn&apos;t configured — copy <code>.env.local.example</code> to{" "}
                    <code>.env.local</code>, fill in your <code>VITE_FIREBASE_*</code> keys, then restart.
                  </div>
                </div>
              )}
            </div>

            <div className="meta-row">
              <div className="cell">
                RECOGNITION ACCURACY
                <b>98.4<span style={{ fontFamily: "var(--mono)", fontSize: 13 }}>%</span></b>
              </div>
              <div className="cell">
                AVG. LATENCY
                <b>312<span style={{ fontFamily: "var(--mono)", fontSize: 13 }}>ms</span></b>
              </div>
              <div className="cell">SYMBOL CLASSES<b>110+</b></div>
              <div className="cell">SOLVERS<b>CNN · Gemini · Ryacas</b></div>
            </div>
          </div>

          <div className="eq-card">
            <span className="marg">FIG. 01 — handwriting → LaTeX</span>
            <span className="corner">B-2024.11</span>
            <div style={{ paddingTop: 28 }}>
              <HandwrittenEquation />
            </div>
            <div className="row2">
              <div className="latex">2x² + 5x − 12 = 0</div>
              <div className="solved">
                x = <span style={{ background: "var(--accent)", padding: "0 4px" }}>3⁄2</span>, −4
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── FEATURES ──────────────────────────────────────────── */}
      <section className="features-section">
        <div className="section-tag">
          <span className="num">02 / WHAT IT DOES</span>
          <span className="line" />
        </div>
        <h2>
          Built for the <span className="it">space between</span><br />
          a notebook and a calculator.
        </h2>

        <div className="feat-grid">
          <div className="feat span4">
            <span className="n">F · 01</span>
            <h3>Handwriting that <span className="it">parses</span></h3>
            <p>
              Scribble naturally — fractions, exponents, summation, integrals.
              Our recognizer handles 110+ symbol classes in mathematical notation.
            </p>
            <div className="icon" style={{ marginTop: 18 }}>
              <svg viewBox="0 0 200 55" width="100%" height={48}>
                <path d="M10 38 C 20 18, 30 46, 40 28 S 60 46, 70 28" stroke="var(--ink)" strokeWidth="2.2" fill="none" strokeLinecap="round"/>
                <text x="85" y="35" fontFamily="JetBrains Mono" fontSize="12" fill="var(--ink-3)">→  \frac{"{1}"}{"{x}"}</text>
              </svg>
            </div>
          </div>

          <div className="feat dark span4">
            <span className="n">F · 02</span>
            <h3>One tap to <span className="it">solve</span></h3>
            <p>
              Convert and solve in a single keystroke. See the answer, the steps,
              and a confidence score. Local CNN first, Gemini fallback.
            </p>
            <div className="icon" style={{ marginTop: 18 }}>
              <div style={{ display: "inline-flex", padding: "8px 14px", background: "var(--accent)", color: "var(--ink)", borderRadius: 99, fontFamily: "var(--mono)", fontSize: 12, letterSpacing: ".06em" }}>
                ⌘ ↵  CONVERT & SOLVE
              </div>
            </div>
          </div>

          <div className="feat accent-bg span4">
            <span className="n">F · 03</span>
            <h3>Boards that <span className="it">remember</span></h3>
            <p>
              Every stroke is versioned. Sign in once and your boards follow you
              across devices — saved to Firestore instantly.
            </p>
            <div className="icon">
              <div style={{ display: "flex", gap: 6, marginTop: 18 }}>
                {[0, 1, 2].map(i => (
                  <div key={i} style={{ flex: 1, height: 32, border: "1px solid var(--ink)", borderRadius: 6, background: `rgba(20,19,15,${i * 0.05})` }} />
                ))}
              </div>
            </div>
          </div>

          <div className="feat span6">
            <span className="n">F · 04</span>
            <h3>Step-by-step <span className="it">derivations</span></h3>
            <p>
              Mathboard doesn&apos;t just give you an answer — it shows the work line by line,
              with the rule applied at each step. Algebra, calculus, linear algebra, ODEs.
            </p>
            <div className="icon" style={{ marginTop: 18, fontFamily: "var(--serif)", fontStyle: "italic", fontSize: 20, color: "var(--ink-2)", lineHeight: 1.4 }}>
              x² + 5x = 14 →{" "}
              <span style={{ color: "var(--ink)" }}>(x+7)(x−2)</span> → x ∈ {"{"}"−7, 2{"}"}
            </div>
          </div>

          <div className="feat span6">
            <span className="n">F · 05</span>
            <h3>Ryacas <span className="it">cross-check</span></h3>
            <p>
              Every Gemini answer is cross-verified against Ryacas (R-based CAS) for symbolic
              accuracy. Mismatches are flagged so you always know the confidence level.
            </p>
            <div className="icon" style={{ marginTop: 18, display: "flex", gap: 8, flexWrap: "wrap" }}>
              {["LOCAL CNN", "GEMINI", "RYACAS"].map(n => (
                <div key={n} style={{ padding: "4px 10px", borderRadius: 99, border: "1px solid var(--ink)", fontFamily: "var(--mono)", fontSize: 10, letterSpacing: ".06em", background: "var(--paper)", color: "var(--ink-2)" }}>{n}</div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
