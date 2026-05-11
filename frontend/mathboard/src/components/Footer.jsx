import { Link } from "react-router-dom";

const GITHUB_URL = "https://github.com/svk6639/mathboard";

export default function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-inner">
        <div className="footer-grid">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div
                style={{
                  width: 28, height: 28,
                  border: "1.5px solid var(--ink)",
                  display: "grid", placeItems: "center",
                  fontFamily: "var(--serif)", fontStyle: "italic", fontSize: 17,
                  flexShrink: 0,
                }}
                aria-hidden="true"
              >
                M
              </div>
              <span style={{ fontWeight: 600, fontSize: 15 }}>Mathboard</span>
            </div>
            <p className="footer-tagline">
              A whiteboard for people who still do math with their hands.
            </p>
          </div>
          <div>
            <h4>Navigate</h4>
            <ul>
              <li><Link to="/" style={{ color: "inherit", textDecoration: "none" }}>Overview</Link></li>
              <li><Link to="/notebooks" style={{ color: "inherit", textDecoration: "none" }}>Boards</Link></li>
              <li><Link to="/dashboard" style={{ color: "inherit", textDecoration: "none" }}>Activity</Link></li>
              <li><Link to="/login" style={{ color: "inherit", textDecoration: "none" }}>Sign in</Link></li>
            </ul>
          </div>
          <div>
            <h4>Stack</h4>
            <ul>
              <li>Local CNN</li>
              <li>Gemini API</li>
              <li>Ryacas (R CAS)</li>
              <li>Firestore</li>
            </ul>
          </div>
          <div>
            <h4>Links</h4>
            <ul>
              <li>
                <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" style={{ color: "inherit" }}>
                  GitHub
                </a>
              </li>
            </ul>
          </div>
        </div>

        <h2 className="footer-big">
          Solve <span className="it">anything</span>—<br />on the same page.
        </h2>

        <div className="footer-bot">
          <span>© 2026 MATHBOARD · Built for math</span>
          <span>v 2.4.1</span>
        </div>
      </div>
    </footer>
  );
}
