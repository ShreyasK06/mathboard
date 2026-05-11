export const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000";

export const SHINY_URL =
  import.meta.env.VITE_SHINY_URL ?? "http://localhost:3838";

/**
 * POSTs the given canvas blob to /convert and parses the response into the
 * shape consumed by ResultPanel. Throws on network failure or non-JSON.
 */
export async function convertCanvasBlob(blob) {
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
  return data;
}
