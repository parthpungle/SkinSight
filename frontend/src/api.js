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

export const getModels = () => request("/models").then((b) => b.models);
export const getMetrics = () => request("/metrics");
export const figureUrl = (name) => `${API_BASE}/figures/${encodeURIComponent(name)}`;

export function predict(file, modelId, includeGradcam = true) {
  const form = new FormData();
  form.append("file", file);
  form.append("model_id", modelId);
  form.append("include_gradcam", String(includeGradcam));
  return request("/predict", { method: "POST", body: form });
}
