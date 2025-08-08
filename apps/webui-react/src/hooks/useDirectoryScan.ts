import { useState, useCallback } from 'react';
import { directoryScanV2Api, generateScanId } from '../services/api/v2/directoryScan';
import type { DirectoryScanResponse } from '../services/api/v2/types';
import { getErrorMessage } from '../utils/errorUtils';

type ScanResult = DirectoryScanResponse;

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

        // Use v2 response shape directly
        setScanResult(response);

        // If there are warnings in the response, show the first one as an error
        if (response.warnings.length > 0 && response.warnings[0] !== 'Scan in progress - connect to WebSocket for real-time updates') {
          setError(response.warnings[0]);
        }
      } catch (err) {
        setError(getErrorMessage(err));
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
