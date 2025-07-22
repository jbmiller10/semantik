import { useState, useCallback } from 'react';
import { directoryScanV2Api, generateScanId } from '../services/api/v2/directoryScan';

interface ScanResult {
  files: string[];
  total_files: number;
  total_size: number;
}

export function useDirectoryScan(scanId?: string) {
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const currentScanId = scanId || generateScanId();

  const startScan = useCallback(
    async (directory: string) => {
      if (!directory || directory.trim() === '') {
        setError('Please enter a directory path');
        return;
      }

      setError(null);
      setScanResult(null);
      setScanning(true);

      try {
        const response = await directoryScanV2Api.preview({
          path: directory,
          scan_id: currentScanId,
          recursive: true,
        });

        // Convert response to legacy format for compatibility
        setScanResult({
          files: response.files.map(f => f.file_path),
          total_files: response.total_files,
          total_size: response.total_size,
        });

        // If there are warnings in the response, show the first one as an error
        if (response.warnings.length > 0 && response.warnings[0] !== 'Scan in progress - connect to WebSocket for real-time updates') {
          setError(response.warnings[0]);
        }
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to scan directory');
        setScanResult(null);
      } finally {
        setScanning(false);
      }
    },
    [currentScanId]
  );

  const reset = useCallback(() => {
    setScanning(false);
    setScanResult(null);
    setError(null);
  }, []);

  return {
    scanning,
    scanResult,
    error,
    startScan,
    reset,
  };
}