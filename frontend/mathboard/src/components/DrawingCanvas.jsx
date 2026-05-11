import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

import { PencilDrawIcon } from "./Icons.jsx";

/**
 * Drawing surface with pen / eraser, undo / redo, and an empty-state hint.
 *
 * Imperative methods exposed via ref:
 *   - getDataUrl()        → string (PNG base64)
 *   - getBlob()           → Promise<Blob>
 *   - loadDataUrl(url)    → void; replaces canvas + resets history
 *   - clear()             → void; fills white + resets content flag
 *   - undo() / redo()     → void
 *
 * Props:
 *   - tool, penSize, eraserSize, penColor: current tool config
 *   - onContentChange(dataUrl): fires after each completed stroke
 *   - onHistoryChange({ canUndo, canRedo, hasContent }): mirrors local state up
 *   - hint: text shown when canvas is empty
 */
const DrawingCanvas = forwardRef(function DrawingCanvas(
  {
    tool,
    penSize,
    eraserSize,
    penColor,
    onContentChange,
    onHistoryChange,
    hint = "Draw a math expression here",
  },
  ref,
) {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const dprRef = useRef(1);
  const isDrawingRef = useRef(false);
  const historyRef = useRef([]);
  const historyIndexRef = useRef(-1);
  const pendingLoadRef = useRef(null); // dataUrl waiting to be drawn after first sizing
  const [hasContent, setHasContent] = useState(false);

  const emitHistory = useCallback(() => {
    onHistoryChange?.({
      canUndo: historyIndexRef.current > 0,
      canRedo: historyIndexRef.current < historyRef.current.length - 1,
      hasContent,
    });
  }, [hasContent, onHistoryChange]);

  const fillWhite = useCallback(() => {
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
    if (!canvas) return null;
    const dataUrl = canvas.toDataURL("image/png");
    const next = historyRef.current.slice(0, historyIndexRef.current + 1);
    next.push(dataUrl);
    if (next.length > 50) next.shift();
    historyRef.current = next;
    historyIndexRef.current = next.length - 1;
    emitHistory();
    return dataUrl;
  }, [emitHistory]);

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

  const drawDataUrl = useCallback(
    (dataUrl) => {
      const canvas = canvasRef.current;
      const ctx = ctxRef.current;
      if (!canvas || !ctx) return;
      const finish = () => {
        // Reset history with the loaded image as the only state.
        const fresh = canvas.toDataURL("image/png");
        historyRef.current = [fresh];
        historyIndexRef.current = 0;
        setHasContent(Boolean(dataUrl));
        emitHistory();
      };
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.restore();
      if (!dataUrl) {
        finish();
        return;
      }
      const img = new Image();
      img.onload = () => {
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        ctx.restore();
        finish();
      };
      img.src = dataUrl;
    },
    [emitHistory],
  );

  // Initial canvas setup + resize handling.
  useEffect(() => {
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

    if (pendingLoadRef.current !== null) {
      drawDataUrl(pendingLoadRef.current);
      pendingLoadRef.current = null;
    } else if (historyRef.current.length === 0) {
      snapshot();
    }

    const onResize = () => setupCanvas(true);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync tool / size / color into context.
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

  const getPos = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const handlePointerDown = (e) => {
    e.preventDefault();
    const ctx = ctxRef.current;
    if (!ctx) return;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* pointer capture may already be taken */
    }
    const { x, y } = getPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 0.01, y + 0.01); // produce a dot on tap
    ctx.stroke();
    isDrawingRef.current = true;
    setHasContent(true);
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
    const dataUrl = snapshot();
    if (dataUrl) onContentChange?.(dataUrl);
  };

  useEffect(() => {
    emitHistory();
  }, [hasContent, emitHistory]);

  useImperativeHandle(
    ref,
    () => ({
      getDataUrl: () => canvasRef.current?.toDataURL("image/png") ?? null,
      getBlob: () =>
        new Promise((resolve, reject) => {
          const canvas = canvasRef.current;
          if (!canvas) return reject(new Error("Canvas unavailable"));
          canvas.toBlob(
            (b) => (b ? resolve(b) : reject(new Error("Could not capture canvas"))),
            "image/png",
          );
        }),
      loadDataUrl: (dataUrl) => {
        if (!ctxRef.current || !canvasRef.current?.width) {
          pendingLoadRef.current = dataUrl ?? null;
          return;
        }
        drawDataUrl(dataUrl);
      },
      clear: () => {
        fillWhite();
        const dataUrl = snapshot();
        setHasContent(false);
        if (dataUrl) onContentChange?.(dataUrl);
      },
      undo: () => {
        if (historyIndexRef.current <= 0) return;
        historyIndexRef.current -= 1;
        restoreIndex(historyIndexRef.current);
        emitHistory();
        const dataUrl = historyRef.current[historyIndexRef.current];
        if (dataUrl) onContentChange?.(dataUrl);
      },
      redo: () => {
        if (historyIndexRef.current >= historyRef.current.length - 1) return;
        historyIndexRef.current += 1;
        restoreIndex(historyIndexRef.current);
        emitHistory();
        const dataUrl = historyRef.current[historyIndexRef.current];
        if (dataUrl) onContentChange?.(dataUrl);
      },
    }),
    [drawDataUrl, emitHistory, fillWhite, onContentChange, restoreIndex, snapshot],
  );

  return (
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
      {!hasContent && (
        <div className="canvas-empty-hint" aria-hidden="true">
          <PencilDrawIcon />
          <span>{hint}</span>
        </div>
      )}
    </div>
  );
});

export default DrawingCanvas;
