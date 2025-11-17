/**
 * WebGPU compatibility shim for Embedding Atlas.
 *
 * Embedding Atlas' WebGPU path currently relies on optional features (such as
 * `shader-f16`) that are not available on all adapters. When these features are
 * missing, the library's internal `requestDevice` call rejects and the
 * visualization never renders.
 *
 * To keep projections reliable across a wide range of browsers and GPUs, this
 * patch intercepts `navigator.gpu.requestAdapter` and forces it to return
 * `null`. This nudges Embedding Atlas onto its WebGL code path, which is more
 * widely supported today.
 *
 * Behaviour:
 * - Runs only in the browser (no-ops on the server or when `navigator.gpu`
 *   is unavailable).
 * - Calls the original `requestAdapter` once for side‑effects, then returns
 *   `null` so WebGPU is effectively disabled for Embedding Atlas.
 * - Sets `window.__embeddingAtlasWebgpuFallback = true` and logs a console
 *   warning so operators can detect that WebGPU was disabled.
 *
 * This is an intentionally conservative default and may be revisited once
 * Embedding Atlas' WebGPU implementation is stable across common hardware.
 */

let patched = false;

export function ensureEmbeddingAtlasWebgpuCompatibility() {
  if (patched || typeof window === 'undefined') {
    return;
  }

  const gpu = (navigator as Navigator & { gpu?: { requestAdapter?: (...args: any[]) => Promise<any> } }).gpu;
  if (!gpu || typeof gpu.requestAdapter !== 'function') {
    patched = true;
    return;
  }

  const originalRequestAdapter = gpu.requestAdapter.bind(gpu);

  gpu.requestAdapter = async function patchedRequestAdapter(...args: any[]) {
    await originalRequestAdapter(...args);
    if (typeof window !== 'undefined') {
      (window as any).__embeddingAtlasWebgpuFallback = true;
      console.warn('[EmbeddingAtlas] Forcing WebGL renderer (WebGPU disabled for compatibility).');
    }
    // Force Embedding Atlas to take its WebGL path by denying WebGPU.
    return null;
  };

  patched = true;
}
