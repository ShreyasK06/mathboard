import { useState } from "react";

import {
  CalculatorIcon,
  DrawCanvasIcon,
  RecognizeIcon,
} from "./Icons.jsx";

const STEPS = [
  {
    Icon: DrawCanvasIcon,
    title: "Draw",
    desc: "Sketch any math expression on the canvas with your mouse or stylus.",
  },
  {
    Icon: RecognizeIcon,
    title: "Recognize",
    desc: "Local CNN handles common symbols instantly. Gemini handles the rest.",
  },
  {
    Icon: CalculatorIcon,
    title: "Solve",
    desc: "Ryacas computes the answer; Gemini solves in parallel and the two are cross-checked.",
  },
];

export default function HowItWorks() {
  const [open, setOpen] = useState(false);
  return (
    <div className="how-section">
      <button
        type="button"
        className="how-toggle"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span>How it works</span>
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className={open ? "how-chevron open" : "how-chevron"}
          aria-hidden="true"
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {open && (
        <div className="how-cards">
          {STEPS.map(({ Icon, title, desc }, i) => (
            <div key={title} style={{ display: "contents" }}>
              <div className="how-card">
                <div className="how-card-icon" aria-hidden="true">
                  <Icon />
                </div>
                <div className="how-card-body">
                  <h4 className="how-card-title">{title}</h4>
                  <p className="how-card-desc">{desc}</p>
                </div>
              </div>
              {i < STEPS.length - 1 && (
                <div className="how-connector" aria-hidden="true">
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M5 12h14M13 6l6 6-6 6" />
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
