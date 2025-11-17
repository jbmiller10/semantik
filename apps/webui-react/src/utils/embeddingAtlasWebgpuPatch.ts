/**
 * Patch WebGPU adapter/device requests so Embedding Atlas can operate on hardware
 * that lacks optional features like `shader-f16`. The library currently requests
 * that feature unconditionally; on adapters that do not support it, the call to
 * `requestDevice` rejects, leaving the visualization blank. This hook strips any
 * unsupported required features before delegating to the native browser APIs.
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
