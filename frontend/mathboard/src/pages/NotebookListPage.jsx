import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { NotebookIcon, PlusIcon, TrashIcon } from "../components/Icons.jsx";
import { useAuth } from "../contexts/AuthContext.jsx";
import {
  createNotebook,
  deleteNotebook,
  listNotebooks,
} from "../services/notes.js";
import { notebooksCache } from "../services/notebooksCache.js";

function formatDate(value) {
  if (!value) return "—";
  const date = value.toDate ? value.toDate() : new Date(value);
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function greeting() {
  const h = new Date().getHours();
  if (h < 12) return "morning";
  if (h < 17) return "afternoon";
  return "evening";
}

export default function NotebookListPage() {
  const { user } = useAuth();
  const navigate = useNavigate();

  // Serve from cache immediately — no loading flash on repeat visits.
  const [notebooks, setNotebooks] = useState(() => notebooksCache.getList(user?.uid) ?? []);
  const [loading, setLoading] = useState(() => notebooksCache.getList(user?.uid) === null);
  const [error, setError] = useState(null);
  const [creating, setCreating] = useState(false);

  const name = (user?.displayName || user?.email || "").split("@")[0] || "there";

  // Fetch from Firebase only when the cache is empty (first visit after login).
  const refresh = useCallback(async () => {
    if (!user) return;
    const cached = notebooksCache.getList(user.uid);
    if (cached) { setNotebooks(cached); setLoading(false); return; }
    setLoading(true);
    try {
      const list = await listNotebooks(user.uid);
      notebooksCache.setList(user.uid, list);
      setNotebooks(list);
      setError(null);
    } catch (err) {
      setError(err?.message ?? "Could not load notebooks");
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => { refresh(); }, [refresh]);

  const handleCreate = async () => {
    if (!user || creating) return;
    setCreating(true);
    try {
      const id = await createNotebook(user.uid);
      // Re-fetch the new notebook doc so we have the server timestamp, then cache it.
      const fresh = await listNotebooks(user.uid);
      notebooksCache.setList(user.uid, fresh);
      navigate(`/notebooks/${id}`);
    } catch (err) {
      setError(err?.message ?? "Could not create notebook");
      setCreating(false);
    }
  };

  const handleDelete = async (notebookId) => {
    if (!user) return;
    const ok = window.confirm("Delete this notebook? This can't be undone.");
    if (!ok) return;
    try {
      await deleteNotebook(user.uid, notebookId);
      notebooksCache.removeFromList(user.uid, notebookId);
      setNotebooks(prev => prev.filter(nb => nb.id !== notebookId));
    } catch (err) {
      setError(err?.message ?? "Could not delete notebook");
    }
  };

  return (
    <main className="notebook-list-page">
      <header className="page-header">
        <div>
          <span className="section-tag" style={{ marginBottom: 10 }}>
            <span className="num">03 — Your boards</span>
            <span className="line" />
          </span>
          <h2 className="page-title">
            Good {greeting()},{" "}
            <span style={{ fontStyle: "italic" }}>{name}.</span>
          </h2>
          <p className="page-subtitle">
            Each notebook is a stack of pages, saved to your account.
          </p>
        </div>
        <button
          type="button"
          className="convert-btn"
          onClick={handleCreate}
          disabled={creating}
        >
          <PlusIcon />
          <span>{creating ? "Creating…" : "New notebook"}</span>
        </button>
      </header>

      {error && (
        <div className="banner banner-error" role="alert">
          <div>{error}</div>
        </div>
      )}

      {loading ? (
        <ul className="notebook-grid" role="list" aria-busy="true" aria-label="Loading notebooks">
          {[0, 1, 2].map((i) => (
            <li key={i} className="notebook-card" style={{ opacity: 1 - i * 0.2 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 14, flex: 1, pointerEvents: "none" }}>
                <div className="notebook-card-mark" style={{ background: "var(--paper-3)" }} />
                <div className="notebook-card-body">
                  <div style={{ height: 14, width: "60%", background: "var(--paper-3)", borderRadius: 3, marginBottom: 6 }} />
                  <div style={{ height: 10, width: "40%", background: "var(--paper-3)", borderRadius: 3 }} />
                </div>
              </div>
            </li>
          ))}
        </ul>
      ) : notebooks.length === 0 ? (
        <div className="empty-state">
          <NotebookIcon width={48} height={48} />
          <h3>No notebooks yet</h3>
          <p>Create your first notebook to start drawing math.</p>
          <button type="button" className="convert-btn" onClick={handleCreate}>
            <PlusIcon />
            <span>New notebook</span>
          </button>
        </div>
      ) : (
        <ul className="notebook-grid" role="list">
          {notebooks.map((nb) => (
            <li key={nb.id} className="notebook-card">
              <Link to={`/notebooks/${nb.id}`} className="notebook-card-link">
                <div className="notebook-card-mark" aria-hidden="true">
                  <NotebookIcon />
                </div>
                <div className="notebook-card-body">
                  <h3 className="notebook-card-title">{nb.title || "Untitled"}</h3>
                  <p className="notebook-card-meta">
                    {(nb.pageCount ?? 1)} page{(nb.pageCount ?? 1) === 1 ? "" : "s"} ·{" "}
                    Updated {formatDate(nb.updatedAt)}
                  </p>
                </div>
              </Link>
              <button
                type="button"
                className="icon-btn danger"
                onClick={() => handleDelete(nb.id)}
                aria-label={`Delete ${nb.title || "notebook"}`}
                title="Delete notebook"
              >
                <TrashIcon />
              </button>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
