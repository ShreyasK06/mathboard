import { useEffect, useRef, useState } from "react";
import "./App.css";
import katex from "katex";
import "katex/dist/katex.min.css";

export default function App() {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const dprRef = useRef(1);

  const [isDrawing, setIsDrawing] = useState(false);

  // Drawing controls
  const [penSize, setPenSize] = useState(6);
  const [eraserSize, setEraserSize] = useState(20);
  const [penColor, setPenColor] = useState("#000000");
  const [tool, setTool] = useState("pen"); // "pen" | "eraser"

  // History for undo/redo
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  // Tab navigation
  const [activeTab, setActiveTab] = useState("solver"); // "solver" | "explorer"

  // Conversion result state
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null); // { latex, solution }
  const [resultError, setResultError] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    const dpr = window.devicePixelRatio || 1;
    dprRef.current = dpr;
    const rect = canvas.getBoundingClientRect();

    canvas.style.width = rect.width + "px";
    canvas.style.height = rect.height + "px";

    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = penColor;
    ctx.lineWidth = penSize;

    ctxRef.current = ctx;

    saveToHistory(ctx);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;

    const currentSize = tool === "eraser" ? eraserSize : penSize;
    ctx.lineWidth = currentSize;

    if (tool === "eraser") {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "#FFFFFF";
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = penColor;
    }
  }, [penSize, eraserSize, penColor, tool]);

  const getPos = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const dpr = dprRef.current || 1;
    return {
      x: (e.clientX - rect.left) * dpr,
      y: (e.clientY - rect.top) * dpr,
    };
  };

  const saveToHistory = (ctxOverride) => {
    const canvas = canvasRef.current;
    const ctx = ctxOverride ?? ctxRef.current;
    if (!canvas || !ctx) return;

    const dataUrl = canvas.toDataURL("image/png");

    setHistory((prev) => {
      const next = prev.slice(0, historyIndex + 1);
      next.push(dataUrl);
      return next;
    });
    setHistoryIndex((prev) => prev + 1);
  };

  const restoreFromHistory = (index) => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;

    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = history[index];
  };

  const handlePointerDown = (e) => {
    e.preventDefault();
    const ctx = ctxRef.current;
    if (!ctx) return;

    e.currentTarget.setPointerCapture(e.pointerId);

    const { x, y } = getPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const handlePointerMove = (e) => {
    if (!isDrawing) return;
    e.preventDefault();

    const ctx = ctxRef.current;
    if (!ctx) return;

    const { x, y } = getPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const handlePointerUp = (e) => {
    if (!isDrawing) return;
    e.preventDefault();
    setIsDrawing(false);
    saveToHistory();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    if (!canvas || !ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    saveToHistory();
  };

  const undo = () => {
    if (historyIndex <= 0) return;
    const newIndex = historyIndex - 1;
    setHistoryIndex(newIndex);
    restoreFromHistory(newIndex);
  };

  const redo = () => {
    if (historyIndex >= history.length - 1) return;
    const newIndex = historyIndex + 1;
    setHistoryIndex(newIndex);
    restoreFromHistory(newIndex);
  };

  const handleConvert = async () => {
    const canvas = canvasRef.current;
    setIsLoading(true);
    setResult(null);
    setResultError(null);

    try {
      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/png")
      );
      const formData = new FormData();
      formData.append("file", blob);

      const res = await fetch("http://127.0.0.1:8000/convert", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      const solutionText =
        data.solution_data?.solution ||
        data.solution_data?.error ||
        "No solution returned";

      setResult({ latex: data.latex, solution: solutionText });
    } catch (err) {
      setResultError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page">
      <h1>MathBoard</h1>

      <nav className="nav-tabs">
        <button
          className={`nav-tab ${activeTab === "solver" ? "active" : ""}`}
          onClick={() => setActiveTab("solver")}
        >
          Math Solver
        </button>
        <button
          className={`nav-tab ${activeTab === "explorer" ? "active" : ""}`}
          onClick={() => setActiveTab("explorer")}
        >
          Dataset Explorer
        </button>
      </nav>

      {activeTab === "solver" && (
        <>
          <div className="toolbar">
            <button onClick={undo} disabled={historyIndex <= 0}>
              Undo
            </button>
            <button onClick={redo} disabled={historyIndex >= history.length - 1}>
              Redo
            </button>
            <button onClick={clearCanvas}>Clear</button>

            <div className="divider" />

            <div className="toolGroup">
              <button
                className={tool === "pen" ? "active" : ""}
                onClick={() => setTool("pen")}
              >
                Pen
              </button>
              <button
                className={`${tool === "eraser" ? "active" : ""} eraser-btn`}
                onClick={() => setTool("eraser")}
              >
                Eraser
              </button>
            </div>

            <div className="divider" />

            <label className="control">
              Thickness
              <input
                type="range"
                min="1"
                max="30"
                value={tool === "eraser" ? eraserSize : penSize}
                onChange={(e) => {
                  const newSize = Number(e.target.value);
                  if (tool === "eraser") setEraserSize(newSize);
                  else setPenSize(newSize);
                }}
              />
              <span className="value">
                {tool === "eraser" ? eraserSize : penSize}px
              </span>
            </label>

            <label className="control">
              Color
              <input
                type="color"
                value={penColor}
                onChange={(e) => {
                  setPenColor(e.target.value);
                  setTool("pen");
                }}
                disabled={tool === "eraser"}
                title={tool === "eraser" ? "Switch to Pen to change color" : ""}
              />
            </label>

            <button
              className="primary"
              onClick={handleConvert}
              disabled={isLoading}
            >
              {isLoading ? "Converting…" : "Convert"}
            </button>
          </div>

          <div className="boardWrap">
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

          {isLoading && (
            <div className="results-panel">
              <div className="loading-spinner">Recognizing and solving…</div>
            </div>
          )}

          {resultError && (
            <div className="results-panel">
              <div className="error-display">Error: {resultError}</div>
            </div>
          )}

          {result && (
            <div className="results-panel">
              <h3>LaTeX</h3>
              <div
                className="latex-display"
                dangerouslySetInnerHTML={{
                  __html: katex.renderToString(result.latex, {
                    throwOnError: false,
                    displayMode: true,
                  }),
                }}
              />
              <h3>Solution</h3>
              <div className="solution-display">{result.solution}</div>
            </div>
          )}
        </>
      )}

      {activeTab === "explorer" && (
        <iframe
          src="http://localhost:3838"
          className="explorer-frame"
          title="HASYv2 Dataset Explorer"
        />
      )}
    </div>
  );
}
