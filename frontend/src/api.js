export const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error("Cannot reach the SkinSight API. Is the backend running?");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Retries while the backend is still starting (it loads all checkpoints at startup). */
export async function withRetry(fn, attempts = 6) {
  let lastError;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (e) {
      lastError = e;
      await sleep(400 * 2 ** i);
    }
  }
  throw lastError;
}

export const getModels = () => withRetry(() => request("/models")).then((b) => b.models);
export const getMetrics = () => withRetry(() => request("/metrics"));
export const getNovelty = () => withRetry(() => request("/novelty"));
export const figureUrl = (name) => `${API_BASE}/figures/${encodeURIComponent(name)}`;

function form(file, fields) {
  const f = new FormData();
  f.append("file", file);
  for (const [k, v] of Object.entries(fields)) f.append(k, String(v));
  return f;
}

export function predict(file, modelId, includeGradcam = true) {
  return request("/predict", {
    method: "POST",
    body: form(file, { model_id: modelId, include_gradcam: includeGradcam }),
  });
}

export function compare(file) {
  return request("/compare", { method: "POST", body: form(file, { include_gradcam: false }) }).then(
    (b) => b.results,
  );
}

export function getSamples(params = {}) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== "" && v !== null && v !== undefined && v !== false) q.set(k, String(v));
  }
  return request(`/samples?${q.toString()}`);
}

export const sampleImageUrl = (id) => `${API_BASE}/samples/${encodeURIComponent(id)}/image`;

/** Fetch a test-set image as a File so it goes through exactly the same /compare path as an upload. */
export async function fetchSampleFile(id) {
  let res;
  try {
    res = await fetch(sampleImageUrl(id));
  } catch {
    throw new Error("Cannot reach the SkinSight API. Is the backend running?");
  }
  if (!res.ok) throw new Error(`Could not load sample ${id}.`);
  const blob = await res.blob();
  return new File([blob], `${id}.jpg`, { type: "image/jpeg" });
}
