import { useCallback, useEffect, useRef, useState } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import "./App.css";

const BACKEND_URL = "http://127.0.0.1:8000";
const SHINY_URL = "http://localhost:3838";

export default function App() {
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
  // eslint-disable-next-line no-unused-vars
  const _tickForUndoRedo = historyTick; // just to keep React reading the ref

  // ---------------------------------------------------------------------
  // Convert
  // ---------------------------------------------------------------------
  const [isLoading, setIsLoading] = useState(false);
  const [convertResult, setConvertResult] = useState(null); // { latex, solution, yacasParsed }
  const [convertError, setConvertError] = useState(null);
  const [hasConverted, setHasConverted] = useState(false);

  const handleConvert = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    setIsLoading(true);
    setConvertResult(null);
    setConvertError(null);
    setHasConverted(true);

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

  // ---------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------
  return (
    <div className="page">
      <header className="page-header">
        <h1>MathBoard</h1>
        <p className="subtitle">Draw math. Let Gemini read it. Let R solve it.</p>
      </header>

      {backendStatus === "down" && (
        <div className="banner-error" role="alert">
          <strong>Backend offline.</strong> Could not reach {BACKEND_URL}.
          Start it with: <code>cd backend &amp;&amp; .venv/Scripts/python -m uvicorn main:app --reload</code>
        </div>
      )}

      {backendStatus === "up" && !geminiKeySet && (
        <div className="banner-warn" role="alert">
          <strong>Gemini API key missing.</strong> Add{" "}
          <code>GEMINI_API_KEY=...</code> to <code>backend/.env</code> and restart the backend.
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
          Dataset Explorer
        </button>
      </nav>

      {activeTab === "solver" && (
        <section className="solver">
          <div className="toolbar">
            <div className="tool-group">
              <button
                type="button"
                className={tool === "pen" ? "tool-btn active" : "tool-btn"}
                onClick={() => setTool("pen")}
              >
                Pen
              </button>
              <button
                type="button"
                className={tool === "eraser" ? "tool-btn active" : "tool-btn"}
                onClick={() => setTool("eraser")}
              >
                Eraser
              </button>
            </div>

            <label className="control">
              <span>Thickness</span>
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
              <span className="value">{tool === "eraser" ? eraserSize : penSize}px</span>
            </label>

            <label className="control">
              <span>Color</span>
              <input
                type="color"
                value={penColor}
                onChange={(e) => {
                  setPenColor(e.target.value);
                  if (tool !== "pen") setTool("pen");
                }}
                disabled={tool === "eraser"}
                title={tool === "eraser" ? "Switch to Pen to pick a color" : "Pick pen color"}
              />
            </label>

            <div className="tool-group">
              <button type="button" className="tool-btn" onClick={undo} disabled={!canUndo}>
                Undo
              </button>
              <button type="button" className="tool-btn" onClick={redo} disabled={!canRedo}>
                Redo
              </button>
              <button type="button" className="tool-btn danger" onClick={clearCanvas}>
                Clear
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
            {isLoading ? "Recognizing..." : "Convert"}
          </button>

          {hasConverted && (
            <div className="results">
              {isLoading && (
                <div className="loading-row">
                  <div className="spinner" aria-hidden="true" />
                  <span>Recognizing handwriting...</span>
                </div>
              )}

              {!isLoading && convertError && (
                <div className="result-error" role="alert">
                  <div className="result-error-title">Could not convert</div>
                  <div className="result-error-body">{convertError}</div>
                </div>
              )}

              {!isLoading && convertResult && (
                <div className="result-success">
                  <h3 className="result-label">Recognized LaTeX</h3>
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

                  {convertResult.operationLabel && (
                    <div className="operation-badge">{convertResult.operationLabel}</div>
                  )}

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

                  <h3 className="result-label">Solution</h3>
                  {convertResult.isSolutionError ? (
                    <div className="result-error">
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
            title="HASYv2 Dataset Explorer"
          />
        </section>
      )}
    </div>
  );
}
