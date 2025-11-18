import type { StartProjectionRequest } from '../types/projection';

export interface ProjectionMetadataHashContext {
  collectionId: string;
  embeddingModel: string;
  vectorCount: number;
  updatedAt?: string | null;
}

function normaliseConfig(config?: Record<string, unknown> | null): Record<string, unknown> | null {
  if (!config) return null;
  const entries = Object.entries(config).sort(([a], [b]) => a.localeCompare(b));
  const normalised: Record<string, unknown> = {};
  for (const [key, value] of entries) {
    normalised[key] = value;
  }
  return normalised;
}

export function buildProjectionMetadataHashPayload(
  context: ProjectionMetadataHashContext,
  request: StartProjectionRequest
): string {
  const reducer = (request.reducer ?? 'umap').toString().toLowerCase();
  const dimensionality = request.dimensionality ?? 2;
  const colorBy = (request.color_by ?? 'document_id').toLowerCase();

  let sampleLimit: number | null = null;
  const aliases: (keyof StartProjectionRequest)[] = ['sample_size', 'sample_n'];
  for (const key of aliases) {
    const raw = request[key];
    if (typeof raw === 'number' && Number.isFinite(raw) && raw > 0) {
      sampleLimit = Math.floor(raw);
      break;
    }
  }

  const payload = {
    collection_id: context.collectionId,
    embedding_model: context.embeddingModel,
    collection_vector_count: context.vectorCount ?? 0,
    collection_updated_at: context.updatedAt ?? null,
    reducer,
    dimensionality,
    color_by: colorBy,
    sample_limit: sampleLimit,
    config: normaliseConfig(request.config ?? null),
  };

  // Keys are inserted in already-sorted order to mirror the backend's
  // json.dumps(sort_keys=True, separators=(',', ':')) canonicalisation.
  return JSON.stringify(payload);
}

export async function computeProjectionMetadataHash(
  context: ProjectionMetadataHashContext,
  request: StartProjectionRequest
): Promise<string> {
  const canonical = buildProjectionMetadataHashPayload(context, request);

  const globalCrypto = (globalThis as typeof globalThis & { crypto?: Crypto }).crypto;
  if (globalCrypto?.subtle && typeof TextEncoder !== 'undefined') {
    const encoder = new TextEncoder();
    const data = encoder.encode(canonical);
    const digest = await globalCrypto.subtle.digest('SHA-256', data);
    const bytes = new Uint8Array(digest);
    let hex = '';
    for (let i = 0; i < bytes.length; i += 1) {
      hex += bytes[i].toString(16).padStart(2, '0');
    }
    return hex;
  }

  // Fallback: non-cryptographic but stable hash. In practice this should not
  // be hit in modern browsers or the test environment, but it keeps behaviour
  // predictable if WebCrypto is unavailable.
  let hash = 0;
  for (let i = 0; i < canonical.length; i += 1) {
    const chr = canonical.charCodeAt(i);
    hash = (hash << 5) - hash + chr;
    hash |= 0;
  }
  return `fallback-${Math.abs(hash)}`;
}
