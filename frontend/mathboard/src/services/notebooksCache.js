// Module-level cache — survives re-renders and navigation, cleared on sign-out.
const _lists = new Map();  // uid -> Notebook[]
const _pages = new Map();  // `${uid}/${notebookId}` -> Page[]

export const notebooksCache = {
  // ── Notebook list ──────────────────────────────────────────────────
  getList(uid)            { return _lists.get(uid) ?? null; },
  setList(uid, list)      { _lists.set(uid, list); },
  updateInList(uid, nb) {
    const list = _lists.get(uid);
    if (list) _lists.set(uid, list.map(x => x.id === nb.id ? nb : x));
  },
  addToList(uid, nb) {
    const list = _lists.get(uid);
    if (list) _lists.set(uid, [nb, ...list]);
  },
  removeFromList(uid, id) {
    const list = _lists.get(uid);
    if (list) _lists.set(uid, list.filter(x => x.id !== id));
  },

  // ── Page content ───────────────────────────────────────────────────
  getPages(uid, notebookId)          { return _pages.get(`${uid}/${notebookId}`) ?? null; },
  setPages(uid, notebookId, pages)   { _pages.set(`${uid}/${notebookId}`, pages); },
  invalidatePages(uid, notebookId)   { _pages.delete(`${uid}/${notebookId}`); },

  clear() { _lists.clear(); _pages.clear(); },
};
