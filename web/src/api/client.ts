export class ApiError extends Error {
  override name = "ApiError";
}

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, options);
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { error: { message: text || res.statusText } };
  }
  if (!res.ok) {
    const payload = data as { error?: { message?: string }; detail?: string };
    throw new ApiError(payload.error?.message || payload.detail || res.statusText);
  }
  return data as T;
}

export async function uploadAsset(file: File): Promise<string> {
  const body = new FormData();
  body.append("file", file);
  const data = await api<{ asset_id: string }>("/v1/assets", { method: "POST", body });
  return data.asset_id;
}
