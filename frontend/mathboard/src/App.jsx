import { useCallback, useEffect, useRef, useState } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import "./App.css";

const BACKEND_URL = "http://127.0.0.1:8000";
const SHINY_URL = "http://localhost:3838";

/* ---------- Inline icons ---------- */
const Icon = {
  Pen: (props) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M12 19l7-7 3 3-7 7-3-3z" />
      <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
      <path d="M2 2l7.586 7.586" />
      <circle cx="11" cy="11" r="2" />
    </svg>
  ),
  Eraser: (props) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M20 20H7L3 16a2 2 0 0 1 0-2.8L13.2 3a2 2 0 0 1 2.8 0L21 8a2 2 0 0 1 0 2.8L11 21" />
      <path d="M14 4l6 6" />
    </svg>
  ),
  Undo: (props) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M3 7v6h6" />
      <path d="M3 13a9 9 0 1 0 3-7L3 9" />
    </svg>
  ),
  Redo: (props) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M21 7v6h-6" />
      <path d="M21 13a9 9 0 1 1-3-7l3 3" />
    </svg>
  ),
  Trash: (props) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M3 6h18" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
    </svg>
  ),
  Sparkles: (props) => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8" />
    </svg>
  ),
  Sun: (props) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
    </svg>
  ),
  Moon: (props) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  ),
  Copy: (props) => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  ),
  Check: (props) => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M20 6L9 17l-5-5" />
    </svg>
  ),
  AlertTriangle: (props) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  ),
  AlertCircle: (props) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  ),
  Logo: (props) => (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M4 19l4-12 3 8 2-5 3 9" />
      <path d="M3 21h18" />
    </svg>
  ),
};

export default function App() {
  // ---------------------------------------------------------------------
  // Theme
  // ---------------------------------------------------------------------
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "light";
    try {
      const saved = window.localStorage.getItem("mathboard-theme");
      if (saved === "light" || saved === "dark") return saved;
      if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
        return "dark";
      }
    } catch {
      /* ignore */
    }
    return "light";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      window.localStorage.setItem("mathboard-theme", theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  // ---------------------------------------------------------------------
  // Backend health
  // ---------------------------------------------------------------------
  const [backendStatus, setBackendStatus] = useState("checking"); // checking | up | down
  const [geminiKeySet, setGeminiKeySet] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/health`, { cache: "no-store" });
        if (!res.ok) throw new Error(`status ${res.status}`);
        const data = await res.json();
        if (cancelled) return;
        setBackendStatus("up");
        setGeminiKeySet(Boolean(data.gemini_key_set));
      } catch {
        if (cancelled) return;
        setBackendStatus("down");
      }
    };

    check();
    return () => {
      cancelled = true;
    };
  }, []);

  // ---------------------------------------------------------------------
  // Tabs
  // ---------------------------------------------------------------------
  const [activeTab, setActiveTab] = useState("solver"); // "solver" | "explorer"

  // ---------------------------------------------------------------------
  // Canvas / drawing state
  // ---------------------------------------------------------------------
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const dprRef = useRef(1);

  const [tool, setTool] = useState("pen"); // "pen" | "eraser"
  const [penSize, setPenSize] = useState(6);
  const [eraserSize, setEraserSize] = useState(24);
  const [penColor, setPenColor] = useState("#111111");

  const isDrawingRef = useRef(false);

  // Undo/redo: snapshots of the canvas as data URLs
  const historyRef = useRef([]);
  const historyIndexRef = useRef(-1);
  const [historyTick, setHistoryTick] = useState(0); // force re-render for disabled state

  const clearToWhite = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    if (!canvas || !ctx) return;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.restore();
  }, []);

  const snapshot = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dataUrl = canvas.toDataURL("image/png");
    const nextHistory = historyRef.current.slice(0, historyIndexRef.current + 1);
    nextHistory.push(dataUrl);
    // Cap history to avoid runaway memory use
    if (nextHistory.length > 50) nextHistory.shift();
    historyRef.current = nextHistory;
    historyIndexRef.current = nextHistory.length - 1;
    setHistoryTick((t) => t + 1);
  }, []);

  const restoreIndex = useCallback((index) => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    if (!canvas || !ctx) return;
    const dataUrl = historyRef.current[index];
    if (!dataUrl) return;
    const img = new Image();
    img.onload = () => {
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      ctx.restore();
    };
    img.src = dataUrl;
  }, []);

  // Initial canvas setup + resize handling
  useEffect(() => {
    if (activeTab !== "solver") return undefined;

    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctxRef.current = ctx;

    const setupCanvas = (preserve = true) => {
      const dpr = window.devicePixelRatio || 1;
      dprRef.current = dpr;

      const rect = canvas.getBoundingClientRect();
      const cssWidth = rect.width;
      const cssHeight = rect.height;

      // Grab existing pixels so a resize doesn't wipe the drawing
      let previous = null;
      if (preserve && canvas.width > 0 && canvas.height > 0) {
        try {
          previous = canvas.toDataURL("image/png");
        } catch {
          previous = null;
        }
      }

      canvas.width = Math.floor(cssWidth * dpr);
      canvas.height = Math.floor(cssHeight * dpr);

      // Reset transforms, then scale so we can draw in CSS pixel coords.
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.scale(dpr, dpr);

      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = penColor;
      ctx.lineWidth = penSize;

      if (previous) {
        const img = new Image();
        img.onload = () => {
          ctx.save();
          ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          ctx.restore();
        };
        img.src = previous;
      }
    };

    setupCanvas(false);

    // Seed history with the blank canvas
    if (historyRef.current.length === 0) {
      snapshot();
    }

    const onResize = () => setupCanvas(true);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  // Sync tool/size/color into the context
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    if (tool === "eraser") {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = eraserSize;
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = penColor;
      ctx.lineWidth = penSize;
    }
  }, [tool, penSize, eraserSize, penColor]);

  // Event handlers
  const getPos = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    // After ctx.scale(dpr, dpr) we draw in CSS pixels, so no DPR multiply here.
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  const handlePointerDown = (e) => {
    e.preventDefault();
    const ctx = ctxRef.current;
    if (!ctx) return;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* some browsers throw if capture already taken */
    }
    const { x, y } = getPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    // Dot on tap
    ctx.lineTo(x + 0.01, y + 0.01);
    ctx.stroke();
    isDrawingRef.current = true;
  };

  const handlePointerMove = (e) => {
    if (!isDrawingRef.current) return;
    e.preventDefault();
    const ctx = ctxRef.current;
    if (!ctx) return;
    const { x, y } = getPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const handlePointerUp = (e) => {
    if (!isDrawingRef.current) return;
    e.preventDefault();
    isDrawingRef.current = false;
    snapshot();
  };

  const clearCanvas = () => {
    clearToWhite();
    snapshot();
  };

  const undo = () => {
    if (historyIndexRef.current <= 0) return;
    historyIndexRef.current -= 1;
    restoreIndex(historyIndexRef.current);
    setHistoryTick((t) => t + 1);
  };

  const redo = () => {
    if (historyIndexRef.current >= historyRef.current.length - 1) return;
    historyIndexRef.current += 1;
    restoreIndex(historyIndexRef.current);
    setHistoryTick((t) => t + 1);
  };

  const canUndo = historyIndexRef.current > 0;
  const canRedo = historyIndexRef.current < historyRef.current.length - 1;
  const _tickForUndoRedo = historyTick; // just to keep React reading the ref
  void _tickForUndoRedo;

  // ---------------------------------------------------------------------
  // Convert
  // ---------------------------------------------------------------------
  const [isLoading, setIsLoading] = useState(false);
  const [convertResult, setConvertResult] = useState(null); // { latex, solution, yacasParsed }
  const [convertError, setConvertError] = useState(null);
  const [hasConverted, setHasConverted] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleConvert = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    setIsLoading(true);
    setConvertResult(null);
    setConvertError(null);
    setHasConverted(true);
    setCopied(false);

    try {
      const blob = await new Promise((resolve, reject) => {
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Could not capture canvas"))), "image/png");
      });

      const formData = new FormData();
      formData.append("file", blob, "board.png");

      const res = await fetch(`${BACKEND_URL}/convert`, {
        method: "POST",
        body: formData,
      });

      let data;
      try {
        data = await res.json();
      } catch {
        throw new Error(`Backend returned a non-JSON response (HTTP ${res.status}).`);
      }

      if (!res.ok && !data?.error) {
        throw new Error(`Backend error: HTTP ${res.status}`);
      }

      if (data.error) {
        setConvertError(data.error);
        return;
      }

      const solData = data.solution_data || {};
      const solutionText =
        solData.solution ||
        solData.error ||
        "No solution returned.";

      setConvertResult({
        latex: data.latex || "",
        solution: solutionText,
        isSolutionError: Boolean(solData.error),
        operationLabel: solData.operation_label || "",
        steps: solData.steps || [],
        latexResult: solData.latex_result || "",
        source: data.source || "gemini",
        confidence: typeof data.confidence === "number" ? data.confidence : null,
        agreement: solData.agreement || "ryacas_unavailable",
        ryacasLatex: (solData.ryacas && solData.ryacas.latex_result) || "",
      });
    } catch (err) {
      const isNetwork =
        err instanceof TypeError || /failed to fetch|networkerror/i.test(err.message);
      setConvertError(
        isNetwork
          ? `Could not reach the backend at ${BACKEND_URL}. Start it with: cd backend && .venv/Scripts/python -m uvicorn main:app --reload`
          : err.message
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopyLatex = async () => {
    if (!convertResult?.latex) return;
    try {
      await navigator.clipboard.writeText(convertResult.latex);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore clipboard errors */
    }
  };

  // ---------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------
  return (
    <div className="page">
      <header className="page-header">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <Icon.Logo />
          </div>
          <div className="brand-text">
            <h1>MathBoard</h1>
            <p className="subtitle">Sketch math by hand. Gemini reads it, SymPy solves it.</p>
          </div>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            title={theme === "dark" ? "Light mode" : "Dark mode"}
          >
            {theme === "dark" ? <Icon.Sun /> : <Icon.Moon />}
          </button>
        </div>
      </header>

      {backendStatus === "down" && (
        <div className="banner banner-error" role="alert">
          <Icon.AlertCircle />
          <div>
            <strong>Backend offline.</strong> Could not reach {BACKEND_URL}. Start it with{" "}
            <code>cd backend &amp;&amp; .venv/Scripts/python -m uvicorn main:app --reload</code>.
          </div>
        </div>
      )}

      {backendStatus === "up" && !geminiKeySet && (
        <div className="banner banner-warn" role="alert">
          <Icon.AlertTriangle />
          <div>
            <strong>Gemini API key missing.</strong> Add{" "}
            <code>GEMINI_API_KEY=...</code> to <code>backend/.env</code> and restart the backend.
          </div>
        </div>
      )}

      <nav className="nav-tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "solver"}
          className={`nav-tab ${activeTab === "solver" ? "active" : ""}`}
          onClick={() => setActiveTab("solver")}
        >
          Math Solver
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "explorer"}
          className={`nav-tab ${activeTab === "explorer" ? "active" : ""}`}
          onClick={() => setActiveTab("explorer")}
        >
          Model & Activity
        </button>
      </nav>

      {activeTab === "solver" && (
        <section className="solver">
          <div className="toolbar">
            <div className="tool-group" role="group" aria-label="Drawing tool">
              <button
                type="button"
                className={tool === "pen" ? "icon-btn active" : "icon-btn"}
                onClick={() => setTool("pen")}
                data-tooltip="Pen"
                aria-label="Pen"
                aria-pressed={tool === "pen"}
              >
                <Icon.Pen />
              </button>
              <button
                type="button"
                className={tool === "eraser" ? "icon-btn active" : "icon-btn"}
                onClick={() => setTool("eraser")}
                data-tooltip="Eraser"
                aria-label="Eraser"
                aria-pressed={tool === "eraser"}
              >
                <Icon.Eraser />
              </button>
            </div>

            <label className="control" title="Stroke thickness">
              <span>Size</span>
              <input
                type="range"
                min="1"
                max="40"
                value={tool === "eraser" ? eraserSize : penSize}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (tool === "eraser") setEraserSize(v);
                  else setPenSize(v);
                }}
              />
              <span className="value">{tool === "eraser" ? eraserSize : penSize}</span>
            </label>

            <label className="control" title={tool === "eraser" ? "Switch to Pen to pick a color" : "Pen color"}>
              <span>Color</span>
              <input
                type="color"
                value={penColor}
                onChange={(e) => {
                  setPenColor(e.target.value);
                  if (tool !== "pen") setTool("pen");
                }}
                disabled={tool === "eraser"}
                aria-label="Pen color"
              />
            </label>

            <div className="toolbar-spacer" />

            <div className="tool-group" role="group" aria-label="History">
              <button
                type="button"
                className="icon-btn"
                onClick={undo}
                disabled={!canUndo}
                data-tooltip="Undo"
                aria-label="Undo"
              >
                <Icon.Undo />
              </button>
              <button
                type="button"
                className="icon-btn"
                onClick={redo}
                disabled={!canRedo}
                data-tooltip="Redo"
                aria-label="Redo"
              >
                <Icon.Redo />
              </button>
              <button
                type="button"
                className="icon-btn danger"
                onClick={clearCanvas}
                data-tooltip="Clear canvas"
                aria-label="Clear canvas"
              >
                <Icon.Trash />
              </button>
            </div>
          </div>

          <div className="board-wrap">
            <canvas
              ref={canvasRef}
              className="board"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
              onPointerLeave={handlePointerUp}
            />
          </div>

          <button
            type="button"
            className="convert-btn"
            onClick={handleConvert}
            disabled={isLoading || backendStatus === "down"}
          >
            {isLoading ? (
              <span className="shimmer-text">Recognizing&hellip;</span>
            ) : (
              <>
                <Icon.Sparkles />
                <span>Convert</span>
              </>
            )}
          </button>

          {hasConverted && (
            <div className="results">
              {isLoading && (
                <div className="result-section" aria-live="polite">
                  <div className="loading-row">
                    <span className="shimmer-text">Recognizing handwriting&hellip;</span>
                  </div>
                  <div className="loading-bar" aria-hidden="true" />
                </div>
              )}

              {!isLoading && convertError && (
                <div className="result-error" role="alert">
                  <Icon.AlertCircle />
                  <div>
                    <div className="result-error-title">Could not convert</div>
                    <div className="result-error-body">{convertError}</div>
                  </div>
                </div>
              )}

              {!isLoading && convertResult && (
                <div className="result-success">
                  <div className="badge-row">
                    {convertResult.operationLabel && (
                      <div className="operation-badge">{convertResult.operationLabel}</div>
                    )}
                    {convertResult.source === "local" ? (
                      <div
                        className="source-badge source-local"
                        title={
                          convertResult.confidence != null
                            ? `Confidence: ${(convertResult.confidence * 100).toFixed(0)}%`
                            : "Recognized locally"
                        }
                      >
                        ⚡ Local model
                      </div>
                    ) : (
                      <div className="source-badge source-gemini" title="Recognized by Gemini">
                        ☁ Gemini
                      </div>
                    )}
                  </div>

                  <div className="result-section">
                    <div className="result-label-row">
                      <h3 className="result-label">Recognized LaTeX</h3>
                      {convertResult.latex && (
                        <button
                          type="button"
                          className={copied ? "copy-btn copied" : "copy-btn"}
                          onClick={handleCopyLatex}
                          aria-label="Copy LaTeX to clipboard"
                        >
                          {copied ? <Icon.Check /> : <Icon.Copy />}
                          <span>{copied ? "Copied" : "Copy LaTeX"}</span>
                        </button>
                      )}
                    </div>
                    <div
                      className="latex-box"
                      dangerouslySetInnerHTML={{
                        __html: katex.renderToString(convertResult.latex || "\\,", {
                          throwOnError: false,
                          displayMode: true,
                        }),
                      }}
                    />
                    <div className="latex-raw">
                      <code>{convertResult.latex}</code>
                    </div>
                  </div>

                  {convertResult.steps && convertResult.steps.length > 0 && (
                    <details className="steps-details">
                      <summary className="steps-summary">
                        Steps ({convertResult.steps.length})
                      </summary>
                      <ol className="steps-list">
                        {convertResult.steps.map((step, i) => (
                          <li key={i}>{step}</li>
                        ))}
                      </ol>
                    </details>
                  )}

                  <div className="result-section">
                    <h3 className="result-label">Solution</h3>
                    {convertResult.isSolutionError ? (
                      <div className="result-error">
                        <Icon.AlertCircle />
                        <div className="result-error-body">{convertResult.solution}</div>
                      </div>
                    ) : convertResult.latexResult ? (
                      <div
                        className="latex-box solution-box"
                        dangerouslySetInnerHTML={{
                          __html: katex.renderToString(convertResult.latexResult, {
                            throwOnError: false,
                            displayMode: true,
                          }),
                        }}
                      />
                    ) : (
                      <div className="solution-text">{convertResult.solution}</div>
                    )}
                  </div>

                  {convertResult.agreement === "match" && (
                    <div className="agreement agreement-match">
                      ✓ Cross-checked with Ryacas (results agree)
                    </div>
                  )}
                  {convertResult.agreement === "differ" && convertResult.ryacasLatex && (
                    <div className="agreement agreement-differ">
                      ⚠ Ryacas got a different answer:&nbsp;
                      <span
                        dangerouslySetInnerHTML={{
                          __html: katex.renderToString(convertResult.ryacasLatex, {
                            throwOnError: false,
                          }),
                        }}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </section>
      )}

      {activeTab === "explorer" && (
        <section className="explorer">
          <iframe
            className="explorer-frame"
            src={SHINY_URL}
            title="MathBoard Model & Activity Dashboard"
          />
        </section>
      )}
    </div>
  );
}
