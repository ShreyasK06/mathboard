import {
  addDoc,
  collection,
  deleteDoc,
  doc,
  getDoc,
  getDocs,
  orderBy,
  query,
  serverTimestamp,
  setDoc,
  updateDoc,
  writeBatch,
} from "firebase/firestore";

import { db } from "../lib/firebase.js";

function notebooksRef(uid) {
  if (!db) throw new Error("Firestore is not configured");
  return collection(db, "users", uid, "notebooks");
}

function notebookRef(uid, notebookId) {
  if (!db) throw new Error("Firestore is not configured");
  return doc(db, "users", uid, "notebooks", notebookId);
}

function pagesRef(uid, notebookId) {
  if (!db) throw new Error("Firestore is not configured");
  return collection(db, "users", uid, "notebooks", notebookId, "pages");
}

export async function listNotebooks(uid) {
  const q = query(notebooksRef(uid), orderBy("updatedAt", "desc"));
  const snap = await getDocs(q);
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
}

export async function createNotebook(uid, title = "Untitled notebook") {
  const ref = await addDoc(notebooksRef(uid), {
    title,
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
    pageCount: 1,
  });
  // Seed with one blank page so the editor always has something to render.
  const firstPage = doc(pagesRef(uid, ref.id));
  await setDoc(firstPage, {
    index: 0,
    dataUrl: null,
    updatedAt: serverTimestamp(),
  });
  return ref.id;
}

export async function renameNotebook(uid, notebookId, title) {
  await updateDoc(notebookRef(uid, notebookId), {
    title,
    updatedAt: serverTimestamp(),
  });
}

export async function deleteNotebook(uid, notebookId) {
  // Subcollection delete: Firestore doesn't cascade. Fetch + batch delete pages.
  const pageDocs = await getDocs(pagesRef(uid, notebookId));
  const batch = writeBatch(db);
  pageDocs.forEach((p) => batch.delete(p.ref));
  batch.delete(notebookRef(uid, notebookId));
  await batch.commit();
}

export async function getNotebook(uid, notebookId) {
  const snap = await getDoc(notebookRef(uid, notebookId));
  if (!snap.exists()) return null;
  return { id: snap.id, ...snap.data() };
}

export async function listPages(uid, notebookId) {
  const q = query(pagesRef(uid, notebookId), orderBy("index", "asc"));
  const snap = await getDocs(q);
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
}

export async function savePage(uid, notebookId, pageId, dataUrl) {
  await Promise.all([
    setDoc(
      doc(pagesRef(uid, notebookId), pageId),
      { dataUrl, updatedAt: serverTimestamp() },
      { merge: true },
    ),
    updateDoc(notebookRef(uid, notebookId), { updatedAt: serverTimestamp() }),
  ]);
}

export async function appendPage(uid, notebookId, index) {
  const ref = doc(pagesRef(uid, notebookId));
  await setDoc(ref, {
    index,
    dataUrl: null,
    updatedAt: serverTimestamp(),
  });
  await updateDoc(notebookRef(uid, notebookId), {
    pageCount: index + 1,
    updatedAt: serverTimestamp(),
  });
  return ref.id;
}

export async function deletePage(uid, notebookId, pageId, remainingPageIds) {
  const batch = writeBatch(db);
  batch.delete(doc(pagesRef(uid, notebookId), pageId));
  // Re-index remaining pages so they stay 0..n-1.
  remainingPageIds.forEach((id, i) => {
    batch.update(doc(pagesRef(uid, notebookId), id), { index: i });
  });
  batch.update(notebookRef(uid, notebookId), {
    pageCount: remainingPageIds.length,
    updatedAt: serverTimestamp(),
  });
  await batch.commit();
}
