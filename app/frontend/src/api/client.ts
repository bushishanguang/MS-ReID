import type {
  ExperimentCurve,
  ExperimentResult,
  RetrievalResponse,
  VisualResult,
} from "../types";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export function mediaUrl(path?: string | null): string {
  if (!path) {
    return "";
  }
  if (path.startsWith("data:")) {
    return path;
  }
  return path.startsWith("http") ? path : `${API_BASE_URL}${path}`;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchVisualImages(): Promise<string[]> {
  const data = await getJson<{ images: string[] }>("/api/visual/images");
  return data.images;
}

export async function fetchVisualResult(imageName: string): Promise<VisualResult> {
  return getJson<VisualResult>(`/api/visual/result?image_name=${encodeURIComponent(imageName)}`);
}

export async function computeVisualResult(file: File): Promise<VisualResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/visual/result`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail ?? `Visual calculation failed: ${response.status}`);
  }
  return response.json() as Promise<VisualResult>;
}

export async function fetchExperimentResults(dataset: string): Promise<ExperimentResult[]> {
  const data = await getJson<{ results: ExperimentResult[] }>(
    `/api/experiments/results?dataset=${encodeURIComponent(dataset)}`,
  );
  return data.results;
}

export async function fetchExperimentCurves(): Promise<ExperimentCurve[]> {
  const data = await getJson<{ curves: ExperimentCurve[] }>("/api/experiments/curves?dataset=market1501");
  return data.curves;
}

export async function searchRetrieval(file: File, topK: number): Promise<RetrievalResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("top_k", String(topK));

  const response = await fetch(`${API_BASE_URL}/api/retrieval/search`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail ?? `Search failed: ${response.status}`);
  }
  return response.json() as Promise<RetrievalResponse>;
}
