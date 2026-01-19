/**
 * DatasetUploadModal - Modal for uploading benchmark datasets
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, X, FileJson, AlertCircle, Loader2 } from 'lucide-react';
import { useUploadDataset, useCreateMapping } from '../../hooks/useBenchmarks';
import { useCollections } from '../../hooks/useCollections';

interface DatasetUploadModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

interface ParsedQuery {
  query_key: string;
  query_text: string;
  relevant_docs?: Array<{ doc_ref: Record<string, unknown>; relevance_grade?: number }>;
}

interface CanonicalDataset {
  schema_version: string;
  queries: ParsedQuery[];
  metadata?: Record<string, unknown>;
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let inQuotes = false;

  const pushField = () => {
    row.push(field);
    field = '';
  };

  const pushRow = () => {
    // Trim and keep rows with at least one non-empty cell
    const cleaned = row.map((cell) => cell.trim());
    if (cleaned.some((cell) => cell.length > 0)) {
      rows.push(cleaned);
    }
    row = [];
  };

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
      continue;
    }
    if (ch === ',') {
      pushField();
      continue;
    }
    if (ch === '\n') {
      pushField();
      pushRow();
      continue;
    }
    if (ch === '\r') {
      continue;
    }
    field += ch;
  }

  pushField();
  pushRow();

  return rows;
}

export function DatasetUploadModal({ onClose, onSuccess }: DatasetUploadModalProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [parsedPreview, setParsedPreview] = useState<ParsedQuery[] | null>(null);
  const [canonicalDataset, setCanonicalDataset] = useState<CanonicalDataset | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedCollectionId, setSelectedCollectionId] = useState<string>('');
  const [errors, setErrors] = useState<Record<string, string>>({});

  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadMutation = useUploadDataset();
  const createMappingMutation = useCreateMapping();
  const { data: collections } = useCollections();

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !uploadMutation.isPending) {
        onClose();
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose, uploadMutation.isPending]);

  const parseFile = useCallback(async (f: File) => {
    setParseError(null);
    setParsedPreview(null);
    setCanonicalDataset(null);

    try {
      const text = await f.text();
      let dataset: CanonicalDataset;

      if (f.name.endsWith('.json')) {
        const raw = JSON.parse(text) as Record<string, unknown>;
        const rawQueries = raw.queries;
        if (!Array.isArray(rawQueries)) {
          throw new Error('Invalid dataset format: missing "queries" array');
        }

        const queries: ParsedQuery[] = rawQueries.map((q, idx) => {
          if (!q || typeof q !== 'object') {
            throw new Error(`Query ${idx + 1} must be an object`);
          }
          const query = q as Record<string, unknown>;
          const query_key = String(query.query_key ?? query.query_id ?? '');
          const query_text = String(query.query_text ?? query.query ?? '');
          if (!query_key || !query_text) {
            throw new Error(`Query ${idx + 1} missing required fields (query_key, query_text)`);
          }

          const rawJudgments = (query.relevant_docs ?? query.relevant_doc_refs ?? []) as unknown;
          if (!Array.isArray(rawJudgments)) {
            throw new Error(`Query ${query_key} has invalid relevant_docs (expected array)`);
          }

          const relevant_docs = rawJudgments
            .filter((j) => j != null)
            .map((j) => {
              if (typeof j === 'string') {
                return { doc_ref: { uri: j }, relevance_grade: 2 };
              }
              if (typeof j === 'object') {
                const obj = j as Record<string, unknown>;
                const grade = typeof obj.relevance_grade === 'number' ? obj.relevance_grade : 2;
                const docRefRaw = obj.doc_ref ?? obj;
                if (typeof docRefRaw === 'string') {
                  return { doc_ref: { uri: docRefRaw }, relevance_grade: grade };
                }
                if (typeof docRefRaw === 'object' && docRefRaw) {
                  return { doc_ref: docRefRaw as Record<string, unknown>, relevance_grade: grade };
                }
              }
              return { doc_ref: { uri: String(j) }, relevance_grade: 2 };
            });

          return { query_key, query_text, relevant_docs };
        });

        dataset = {
          schema_version: String(raw.schema_version ?? '1.0'),
          queries,
          metadata: (raw.metadata as Record<string, unknown> | undefined) ?? undefined,
        };
      } else if (f.name.endsWith('.csv')) {
        const rows = parseCsv(text);
        if (rows.length === 0) {
          throw new Error('CSV contains no rows');
        }

        const header = rows[0].map((c) => c.trim().toLowerCase());
        const hasHeader = header.includes('query_key') || header.includes('query_text') || header.includes('query');
        const startIdx = hasHeader ? 1 : 0;

        const queries: ParsedQuery[] = [];
        for (let i = startIdx; i < rows.length; i++) {
          const row = rows[i];
          if (row.length < 2) continue;
          const query_key = row[0]?.trim();
          const query_text = row[1]?.trim();
          if (!query_key || !query_text) continue;

          const refs = row.slice(2).map((r) => r.trim()).filter((r) => r.length > 0);
          const relevant_docs = refs.map((ref) => ({ doc_ref: { uri: ref }, relevance_grade: 2 }));

          queries.push({ query_key, query_text, relevant_docs });
        }

        if (queries.length === 0) {
          throw new Error('CSV contains no valid queries (expected at least query_key and query_text)');
        }

        dataset = { schema_version: '1.0', queries };
      } else {
        throw new Error('Unsupported file format. Please use JSON or CSV.');
      }

      setCanonicalDataset(dataset);
      setParsedPreview(dataset.queries.slice(0, 10));
    } catch (err) {
      setParseError(err instanceof Error ? err.message : 'Failed to parse file');
    }
  }, []);

  const handleFileSelect = useCallback((f: File) => {
    setFile(f);
    parseFile(f);

    // Auto-fill name from filename if empty
    if (!name) {
      const baseName = f.name.replace(/\.(json|csv)$/i, '');
      setName(baseName);
    }
  }, [name, parseFile]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && (droppedFile.name.endsWith('.json') || droppedFile.name.endsWith('.csv'))) {
      handleFileSelect(droppedFile);
    } else {
      setParseError('Please drop a JSON or CSV file');
    }
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const validateForm = () => {
    const newErrors: Record<string, string> = {};

    if (!name.trim()) {
      newErrors.name = 'Dataset name is required';
    }

    if (!file) {
      newErrors.file = 'Please select a file to upload';
    }

    if (parseError) {
      newErrors.file = parseError;
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm() || !file || !canonicalDataset) {
      return;
    }

    try {
      const baseName = name.trim() || file.name.replace(/\.(json|csv)$/i, '') || 'dataset';
      const jsonBlob = new Blob([JSON.stringify(canonicalDataset)], { type: 'application/json' });
      const uploadFile = new File([jsonBlob], `${baseName}.json`, { type: 'application/json' });

      const result = await uploadMutation.mutateAsync({
        data: { name: name.trim(), description: description.trim() || undefined },
        file: uploadFile,
      });

      // Optionally create mapping if collection selected
      if (selectedCollectionId) {
        await createMappingMutation.mutateAsync({
          datasetId: result.id,
          data: { collection_id: selectedCollectionId },
        });
      }

      onSuccess();
    } catch {
      // Error handled by mutation
    }
  };

  const isSubmitting = uploadMutation.isPending || createMappingMutation.isPending;

  return (
    <div className="fixed inset-0 bg-black/50 dark:bg-black/80 flex items-center justify-center p-4 z-50">
      <div className="panel w-full max-w-xl max-h-[90vh] overflow-y-auto rounded-2xl shadow-2xl">
        {/* Header */}
        <div className="px-6 py-5 border-b border-[var(--border)] flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-[var(--text-primary)]">Upload Dataset</h3>
            <p className="mt-1 text-sm text-[var(--text-muted)]">
              Upload a benchmark dataset with queries and ground truth
            </p>
          </div>
          <button
            onClick={onClose}
            disabled={isSubmitting}
            className="p-2 text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded-lg transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="px-6 py-4 space-y-4">
            {/* Name Input */}
            <div>
              <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Dataset Name <span className="text-red-400">*</span>
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                disabled={isSubmitting}
                className={`w-full px-4 py-2.5 input-field rounded-xl ${errors.name ? 'border-red-500/50' : ''}`}
                placeholder="My Benchmark Dataset"
              />
              {errors.name && (
                <p className="mt-1 text-sm text-red-400">{errors.name}</p>
              )}
            </div>

            {/* Description Input */}
            <div>
              <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Description
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                disabled={isSubmitting}
                rows={2}
                className="w-full px-4 py-2.5 input-field rounded-xl"
                placeholder="Optional description..."
              />
            </div>

            {/* File Drop Zone */}
            <div>
              <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Dataset File <span className="text-red-400">*</span>
              </label>
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
                className={`
                  border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors
                  ${isDragging
                    ? 'border-gray-400 dark:border-white bg-[var(--bg-tertiary)]'
                    : 'border-[var(--border)] hover:border-[var(--border-strong)] hover:bg-[var(--bg-tertiary)]'
                  }
                  ${errors.file ? 'border-red-500/50' : ''}
                `}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json,.csv"
                  onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                  className="hidden"
                />
                {file ? (
                  <div className="flex items-center justify-center gap-3">
                    <FileJson className="h-8 w-8 text-[var(--text-muted)]" />
                    <div className="text-left">
                      <p className="font-medium text-[var(--text-primary)]">{file.name}</p>
                      <p className="text-sm text-[var(--text-muted)]">
                        {(file.size / 1024).toFixed(1)} KB
                        {canonicalDataset && ` • ${canonicalDataset.queries.length} queries`}
                      </p>
                    </div>
                  </div>
                ) : (
                  <>
                    <Upload className="mx-auto h-8 w-8 text-[var(--text-muted)]" />
                    <p className="mt-2 text-sm text-[var(--text-secondary)]">
                      Drag and drop a file here, or click to browse
                    </p>
                    <p className="mt-1 text-xs text-[var(--text-muted)]">
                      Supports JSON and CSV formats
                    </p>
                  </>
                )}
              </div>
              {errors.file && (
                <p className="mt-1 text-sm text-red-400">{errors.file}</p>
              )}
            </div>

            {/* Parse Error */}
            {parseError && (
              <div className="flex items-start gap-2 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-400">{parseError}</p>
              </div>
            )}

            {/* Preview */}
            {parsedPreview && parsedPreview.length > 0 && (
              <div>
                <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                  Preview (first {parsedPreview.length} queries)
                </label>
                <div className="max-h-40 overflow-y-auto bg-[var(--bg-tertiary)] rounded-lg border border-[var(--border)]">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-[var(--bg-tertiary)]">
                      <tr className="border-b border-[var(--border)]">
                        <th className="px-3 py-2 text-left text-xs font-medium text-[var(--text-muted)]">Key</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-[var(--text-muted)]">Query</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-[var(--text-muted)]">Docs</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--border)]">
                      {parsedPreview.map((q, idx) => (
                        <tr key={idx}>
                          <td className="px-3 py-2 text-[var(--text-muted)] font-mono text-xs">
                            {q.query_key}
                          </td>
                          <td className="px-3 py-2 text-[var(--text-secondary)] truncate max-w-[200px]">
                            {q.query_text}
                          </td>
                          <td className="px-3 py-2 text-[var(--text-muted)]">
                            {q.relevant_docs?.length ?? 0}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Optional Collection Mapping */}
            <div>
              <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Link to Collection (Optional)
              </label>
              <select
                value={selectedCollectionId}
                onChange={(e) => setSelectedCollectionId(e.target.value)}
                disabled={isSubmitting}
                className="w-full px-4 py-2.5 input-field rounded-xl"
              >
                <option value="">Skip - I'll map later</option>
                {collections?.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-xs text-[var(--text-muted)]">
                Create a mapping to a collection after upload
              </p>
            </div>
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-[var(--border)] flex justify-end gap-3">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="px-4 py-2 text-sm font-medium text-[var(--text-secondary)] border border-[var(--border)] rounded-xl hover:bg-[var(--bg-tertiary)] transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !file || !canonicalDataset || !!parseError}
              className="px-6 py-2 text-sm font-bold text-gray-900 bg-gray-200 dark:bg-white rounded-xl hover:bg-gray-300 dark:hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Uploading...
                </span>
              ) : (
                'Upload Dataset'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
