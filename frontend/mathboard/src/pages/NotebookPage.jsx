import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import BackendStatusBanner from "../components/BackendStatusBanner.jsx";
import DrawingCanvas from "../components/DrawingCanvas.jsx";
import {
  ChevronLeftIcon,
  EraserIcon,
  PenIcon,
  RedoIcon,
  SparklesIcon,
  TrashIcon,
  UndoIcon,
} from "../components/Icons.jsx";
import PageNavigator from "../components/PageNavigator.jsx";
import ResultPanel from "../components/ResultPanel.jsx";
import { useAuth } from "../contexts/AuthContext.jsx";
import { useBackendHealth } from "../hooks/useBackendHealth.js";
import { BACKEND_URL, convertCanvasBlob } from "../services/backend.js";
import {
  appendPage,
  deletePage,
  getNotebook,
  listPages,
  renameNotebook,
  savePage,
} from "../services/notes.js";
import { notebooksCache } from "../services/notebooksCache.js";

const DEFAULT_TITLE = "Untitled notebook";

export default function NotebookPage() {
  const { id: notebookId } = useParams();
  const { user } = useAuth();
  const { status: backendStatus, geminiKeySet } = useBackendHealth();

  const canvasRef = useRef(null);

  const [notebook, setNotebook] = useState(null);
  const [pages, setPages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loadingNotebook, setLoadingNotebook] = useState(true);
  const [loadingPages, setLoadingPages] = useState(true);
  const [notebookError, setNotebookError] = useState(null);

  // Save state — manual save replaces auto-save
  const [isDirty, setIsDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const dirtyPageIds = useRef(new Set());

  const [tool, setTool] = useState("pen");
  const [penSize] = useState(6);
  const [eraserSize] = useState(24);
  const [penColor, setPenColor] = useState("#111111");
  const [history, setHistory] = useState({ canUndo: false, canRedo: false, hasContent: false });

  const [isConverting, setIsConverting] = useState(false);
  const [convertResult, setConvertResult] = useState(null);
  const [convertError, setConvertError] = useState(null);
  const [hasConverted, setHasConverted] = useState(false);
  const [copied, setCopied] = useState(false);

  const [titleDraft, setTitleDraft] = useState("");

  // ── Initial load ──────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    if (!user || !notebookId) return undefined;
    (async () => {
      setLoadingNotebook(true);
      setLoadingPages(true);
      try {
        // Phase 1: notebook metadata (fast — usually already in cache)
        const nb = await getNotebook(user.uid, notebookId);
        if (cancelled) return;
        if (!nb) { setNotebookError("Notebook not found."); setLoadingNotebook(false); return; }
        setNotebook(nb);
        setTitleDraft(nb.title || DEFAULT_TITLE);
        setLoadingNotebook(false);

        // Phase 2: pages — serve from cache if available, otherwise fetch
        const cached = notebooksCache.getPages(user.uid, notebookId);
        if (cached) {
          setPages(cached);
          setCurrentIndex(0);
          setLoadingPages(false);
          return;
        }
        const ps = await listPages(user.uid, notebookId);
        if (cancelled) return;
        notebooksCache.setPages(user.uid, notebookId, ps);
        setPages(ps);
        setCurrentIndex(0);
      } catch (err) {
        if (!cancelled) setNotebookError(err?.message ?? "Could not load notebook");
      } finally {
        if (!cancelled) { setLoadingNotebook(false); setLoadingPages(false); }
      }
    })();
    return () => { cancelled = true; };
  }, [user, notebookId]);

  // ── Load page content into canvas when index or pages change ──────────
  useEffect(() => {
    if (!pages.length || loadingPages) return;
    const page = pages[currentIndex];
    canvasRef.current?.loadDataUrl(page?.dataUrl ?? null);
    setHasConverted(false);
    setConvertResult(null);
    setConvertError(null);
  }, [pages, currentIndex, loadingPages]);

  // ── Warn before leaving with unsaved changes ──────────────────────────
  useEffect(() => {
    const onBeforeUnload = (e) => {
      if (!isDirty) return;
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [isDirty]);

  // ── Content change: update local state only, mark dirty ──────────────
  const handleContentChange = useCallback((dataUrl) => {
    setPages(prev => prev.map((p, i) => (i === currentIndex ? { ...p, dataUrl } : p)));
    if (pages[currentIndex]?.id) {
      dirtyPageIds.current.add(pages[currentIndex].id);
    }
    setIsDirty(true);
  }, [currentIndex, pages]);

  // ── Page navigation: capture canvas → local state, no Firebase write ──
  const goToPage = useCallback((nextIndex) => {
    if (nextIndex < 0 || nextIndex >= pages.length) return;
    // Capture latest canvas state before switching
    const dataUrl = canvasRef.current?.getDataUrl();
    if (dataUrl && pages[currentIndex]) {
      setPages(prev => prev.map((p, i) => i === currentIndex ? { ...p, dataUrl } : p));
    }
    setCurrentIndex(nextIndex);
  }, [pages, currentIndex]);

  // ── Manual save ───────────────────────────────────────────────────────
  const handleSave = useCallback(async () => {
    if (!user || !notebook || isSaving) return;
    // Capture the very latest canvas content for the current page
    const currentDataUrl = canvasRef.current?.getDataUrl();
    const pagesToSave = pages.map((p, i) => {
      if (i === currentIndex && currentDataUrl) return { ...p, dataUrl: currentDataUrl };
      return p;
    });

    setIsSaving(true);
    try {
      // Write only pages that have been modified since the last save
      const writes = pagesToSave
        .filter(p => dirtyPageIds.current.has(p.id) || (p === pagesToSave[currentIndex] && currentDataUrl))
        .map(p => savePage(user.uid, notebook.id, p.id, p.dataUrl ?? ""));

      if (writes.length === 0) { setIsSaving(false); setIsDirty(false); return; }
      await Promise.all(writes);

      // Re-fetch ONLY this notebook's pages from Firebase to confirm and refresh cache
      const fresh = await listPages(user.uid, notebookId);
      notebooksCache.setPages(user.uid, notebookId, fresh);
      // Also update the updatedAt in the notebooks list cache
      const nbSnap = await getNotebook(user.uid, notebookId);
      if (nbSnap) notebooksCache.updateInList(user.uid, nbSnap);

      setPages(fresh);
      dirtyPageIds.current.clear();
      setIsDirty(false);
    } catch (err) {
      setNotebookError(err?.message ?? "Save failed");
    } finally {
      setIsSaving(false);
    }
  }, [user, notebook, notebookId, pages, currentIndex, isSaving]);

  // ── Add page ──────────────────────────────────────────────────────────
  const handleAddPage = useCallback(async () => {
    if (!user || !notebook) return;
    const nextIndex = pages.length;
    try {
      const newId = await appendPage(user.uid, notebook.id, nextIndex);
      const newPages = [...pages, { id: newId, index: nextIndex, dataUrl: null }];
      notebooksCache.setPages(user.uid, notebookId, newPages);
      setPages(newPages);
      setCurrentIndex(nextIndex);
    } catch (err) {
      setNotebookError(err?.message ?? "Could not add page");
    }
  }, [user, notebook, notebookId, pages]);

  // ── Delete page ───────────────────────────────────────────────────────
  const handleDeletePage = useCallback(async () => {
    if (!user || !notebook || pages.length <= 1) return;
    const ok = window.confirm("Delete this page?");
    if (!ok) return;
    const remaining = pages.filter((_, i) => i !== currentIndex);
    try {
      await deletePage(user.uid, notebook.id, pages[currentIndex].id, remaining.map(p => p.id));
      const updated = remaining.map((p, i) => ({ ...p, index: i }));
      notebooksCache.setPages(user.uid, notebookId, updated);
      setPages(updated);
      setCurrentIndex(Math.max(0, currentIndex - 1));
    } catch (err) {
      setNotebookError(err?.message ?? "Could not delete page");
    }
  }, [user, notebook, notebookId, pages, currentIndex]);

  // ── Title rename ──────────────────────────────────────────────────────
  const commitTitle = useCallback(async () => {
    if (!user || !notebook) return;
    const trimmed = titleDraft.trim() || DEFAULT_TITLE;
    if (trimmed === (notebook.title || DEFAULT_TITLE)) return;
    try {
      await renameNotebook(user.uid, notebook.id, trimmed);
      const updated = { ...notebook, title: trimmed };
      setNotebook(updated);
      notebooksCache.updateInList(user.uid, updated);
    } catch (err) {
      setNotebookError(err?.message ?? "Rename failed");
    }
  }, [user, notebook, titleDraft]);

  // ── Convert & solve ───────────────────────────────────────────────────
  const handleConvert = async () => {
    if (!canvasRef.current) return;
    setIsConverting(true);
    setConvertResult(null);
    setConvertError(null);
    setHasConverted(true);
    setCopied(false);
    try {
      const blob = await canvasRef.current.getBlob();
      const data = await convertCanvasBlob(blob);
      if (data.error) { setConvertError(data.error); return; }
      const solData = data.solution_data || {};
      setConvertResult({
        latex: data.latex || "",
        solution: solData.solution || solData.error || "No solution returned.",
        isSolutionError: Boolean(solData.error),
        operationLabel: solData.operation_label || "",
        steps: solData.steps || [],
        latexResult: solData.latex_result || "",
        source: data.source || "gemini",
        confidence: typeof data.confidence === "number" ? data.confidence : null,
        agreement: solData.agreement || "crosscheck_unavailable",
        primarySolver: solData.primary_solver || "ryacas",
        crosscheckLatex: (solData.crosscheck && solData.crosscheck.latex_result) || "",
      });
    } catch (err) {
      const isNetwork = err instanceof TypeError || /failed to fetch|networkerror/i.test(err.message);
      setConvertError(
        isNetwork
          ? `Could not reach the backend at ${BACKEND_URL}. Start it with: cd backend && .venv/Scripts/python -m uvicorn main:app --reload`
          : err.message,
      );
    } finally {
      setIsConverting(false);
    }
  };

  const handleCopyLatex = async () => {
    if (!convertResult?.latex) return;
    try {
      await navigator.clipboard.writeText(convertResult.latex);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* ignore */ }
  };

  // ── Loading / error screens ───────────────────────────────────────────
  if (loadingNotebook) {
    return (
      <main className="notebook-page">
        <div className="page-loading">
          <span className="shimmer-text">Loading notebook…</span>
        </div>
      </main>
    );
  }

  if (notebookError && !notebook) {
    return (
      <main className="notebook-page">
        <div className="banner banner-error" role="alert"><div>{notebookError}</div></div>
        <Link to="/notebooks" className="back-link"><ChevronLeftIcon /> Back to notebooks</Link>
      </main>
    );
  }

  // ── Main 3-column layout ──────────────────────────────────────────────
  return (
    <div className="nb-layout">
      {/* ── Left rail ── */}
      <aside className="nb-rail">
        <button
          type="button"
          className={`rail-btn${tool === "pen" ? " active" : ""}`}
          onClick={() => setTool("pen")}
          data-tooltip="Pen"
          aria-label="Pen tool"
        >
          <PenIcon />
        </button>
        <button
          type="button"
          className={`rail-btn${tool === "eraser" ? " active" : ""}`}
          onClick={() => setTool("eraser")}
          data-tooltip="Eraser"
          aria-label="Eraser tool"
        >
          <EraserIcon />
        </button>

        <div className="nb-rail-sep" />

        <button
          type="button"
          className="rail-btn"
          onClick={() => canvasRef.current?.undo()}
          disabled={!history.canUndo}
          data-tooltip="Undo"
          aria-label="Undo"
        >
          <UndoIcon />
        </button>
        <button
          type="button"
          className="rail-btn"
          onClick={() => canvasRef.current?.redo()}
          disabled={!history.canRedo}
          data-tooltip="Redo"
          aria-label="Redo"
        >
          <RedoIcon />
        </button>

        <div className="nb-rail-sep" />

        <button
          type="button"
          className="rail-btn danger"
          onClick={() => canvasRef.current?.clear()}
          data-tooltip="Clear page"
          aria-label="Clear page"
        >
          <TrashIcon />
        </button>

        <div style={{ flex: 1 }} />

        <input
          type="color"
          className="rail-color"
          value={penColor}
          onChange={(e) => setPenColor(e.target.value)}
          title="Pen color"
          aria-label="Pen color"
        />
      </aside>

      {/* ── Center canvas ── */}
      <div className="nb-canvas-area">
        <BackendStatusBanner status={backendStatus} geminiKeySet={geminiKeySet} />

        {notebookError && notebook && (
          <div className="banner banner-warn" role="alert">
            <div>{notebookError}</div>
          </div>
        )}

        <div className="nb-title-bar">
          <Link to="/notebooks" className="back-link" style={{ flexShrink: 0 }}>
            <ChevronLeftIcon />
          </Link>
          <input
            className="nb-title-input"
            value={titleDraft}
            onChange={(e) => setTitleDraft(e.target.value)}
            onBlur={commitTitle}
            onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }}
            aria-label="Notebook title"
            maxLength={120}
          />
          {/* Save button */}
          <button
            type="button"
            onClick={handleSave}
            disabled={isSaving || !isDirty}
            style={{
              flexShrink: 0,
              padding: "4px 12px",
              borderRadius: 99,
              border: isDirty ? "1px solid var(--ink)" : "1px solid var(--rule)",
              background: isDirty ? "var(--ink)" : "transparent",
              color: isDirty ? "var(--paper)" : "var(--ink-4)",
              fontFamily: "var(--mono)",
              fontSize: 11,
              letterSpacing: ".06em",
              cursor: isDirty ? "pointer" : "default",
              transition: "all .15s",
              whiteSpace: "nowrap",
            }}
            aria-label="Save notebook"
          >
            {isSaving ? "Saving…" : isDirty ? "● Save" : "✓ Saved"}
          </button>
        </div>

        <div className="nb-canvas-wrap">
          {loadingPages ? (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", fontFamily: "var(--mono)", fontSize: 12, letterSpacing: ".06em" }}>
              <span className="shimmer-text">Loading pages…</span>
            </div>
          ) : (
            <DrawingCanvas
              ref={canvasRef}
              tool={tool}
              penSize={penSize}
              eraserSize={eraserSize}
              penColor={penColor}
              onContentChange={handleContentChange}
              onHistoryChange={setHistory}
              hint={`Page ${currentIndex + 1}`}
            />
          )}
        </div>

        <div className="nb-page-nav">
          <PageNavigator
            currentIndex={currentIndex}
            pageCount={pages.length}
            onPrev={() => goToPage(currentIndex - 1)}
            onNext={() => goToPage(currentIndex + 1)}
            onAdd={handleAddPage}
            onDelete={handleDeletePage}
            saving={isSaving}
          />
        </div>
      </div>

      {/* ── Right solver ── */}
      <aside className="nb-solver">
        <div className="nb-solver-header">Solution</div>

        <button
          type="button"
          className="convert-btn"
          onClick={handleConvert}
          disabled={isConverting || backendStatus === "down"}
          style={{ width: "100%" }}
        >
          {isConverting ? (
            <span className="shimmer-text">Recognizing&hellip;</span>
          ) : (
            <>
              <SparklesIcon />
              <span>Convert &amp; Solve</span>
            </>
          )}
        </button>

        {hasConverted && (
          <ResultPanel
            isLoading={isConverting}
            error={convertError}
            result={convertResult}
            copied={copied}
            onCopyLatex={handleCopyLatex}
          />
        )}
      </aside>
    </div>
  );
}
