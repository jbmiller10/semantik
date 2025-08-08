import { useState, useCallback, useRef } from 'react';
import { useWebSocket } from './useWebSocket';
import { directoryScanV2Api, generateScanId } from '../services/api/v2/directoryScan';
import type { DirectoryScanProgress, DirectoryScanResponse } from '../services/api/v2/types';
import { getErrorMessage } from '../utils/errorUtils';

type ScanResult = Pick<DirectoryScanResponse, 'files' | 'total_files' | 'total_size' | 'warnings'>;

interface ScanProgress {
  current_path?: string;
  files_scanned?: number;
  total_files?: number;
  percentage?: number;
}

export function useDirectoryScanWebSocket(scanId?: string) {
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [scanProgress, setScanProgress] = useState<ScanProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useWebSocketMode, setUseWebSocketMode] = useState(true);
  const scanIdRef = useRef<string>(scanId || generateScanId());

  // WebSocket URL for v2 API
  const wsUrl = scanning && useWebSocketMode 
    ? directoryScanV2Api.getWebSocketUrl(scanIdRef.current)
    : null;

  const { disconnect } = useWebSocket(wsUrl, {
    onOpen: () => {
      console.log('Directory scan WebSocket connected');
    },
    onMessage: (event) => {
      try {
        const message: DirectoryScanProgress = JSON.parse(event.data);
        console.log('Directory scan WebSocket message:', message);

        switch (message.type) {
          case 'started':
            setScanProgress({ files_scanned: 0 });
            break;

          case 'counting':
            setScanProgress({
              current_path: message.data.message || 'Counting files...',
              files_scanned: 0,
            });
            break;

          case 'progress':
            setScanProgress({
              current_path: message.data.current_path,
              files_scanned: message.data.files_scanned,
              total_files: message.data.total_files,
              percentage: message.data.percentage,
            });
            break;

          case 'completed':
            setScanning(false);
            // Completed message may not include files. Preserve files from initial preview.
            setScanResult(prev => ({
              files: prev?.files ?? [],
              total_files: message.data.total_files || prev?.total_files || 0,
              total_size: message.data.total_size || prev?.total_size || 0,
              warnings: message.data.warnings || prev?.warnings || [],
            }));
            setScanProgress(null);
            break;
            
          case 'warning':
            // Handle warning messages during scan
            if (message.data.message) {
              setScanResult(prev => prev ? {
                ...prev,
                warnings: [...(prev.warnings || []), message.data.message || ''],
              } : null);
            }
            break;

          case 'error':
            setScanning(false);
            setError(message.data.message || 'Scan failed');
            setScanProgress(null);
            break;
        }
      } catch (error) {
        console.error('Error parsing directory scan WebSocket message:', error);
      }
    },
    onError: () => {
      console.error('Directory scan WebSocket error, falling back to REST API');
      setUseWebSocketMode(false);
    },
    onClose: () => {
      console.log('Directory scan WebSocket closed');
      if (scanning) {
        // If we were still scanning when the connection closed, mark as complete
        setScanning(false);
      }
    },
  });

  const startScanREST = useCallback(
    async (directory: string) => {
      setError(null);
      setScanResult(null);
      setScanProgress(null);
      setScanning(true);

      try {
        const response = await directoryScanV2Api.preview({
          path: directory,
          scan_id: scanIdRef.current,
          recursive: true,
        });

        // Use v2 response shape directly
        setScanResult({
          files: response.files,
          total_files: response.total_files,
          total_size: response.total_size,
          warnings: response.warnings.filter(w => w !== 'Scan in progress - connect to WebSocket for real-time updates'),
        });
      } catch (err) {
        setError(getErrorMessage(err));
        setScanResult(null);
      } finally {
        setScanning(false);
      }
    },
    []
  );

  const startScan = useCallback(
    async (directory: string) => {
      if (!directory || directory.trim() === '') {
        setError('Please enter a directory path');
        return;
      }

      // Generate new scan ID for each scan
      scanIdRef.current = generateScanId();

      // If WebSocket mode failed previously, use REST directly
      if (!useWebSocketMode) {
        return startScanREST(directory);
      }

      // Try WebSocket mode - start API call which will trigger WebSocket updates
      setError(null);
      setScanResult(null);
      setScanProgress(null);
      setScanning(true);

      try {
        // Start the scan via API (this triggers WebSocket updates)
        const response = await directoryScanV2Api.preview({
          path: directory,
          scan_id: scanIdRef.current,
          recursive: true,
        });

        // If we get an immediate response (small directory), use it
        if (response.files.length > 0 || !response.warnings.includes('Scan in progress - connect to WebSocket for real-time updates')) {
          setScanning(false);
          setScanResult({
            files: response.files,
            total_files: response.total_files,
            total_size: response.total_size,
            warnings: response.warnings.filter(w => w !== 'Scan in progress - connect to WebSocket for real-time updates'),
          });
        }
        // Otherwise, WebSocket updates will handle the progress
      } catch (err) {
        setScanning(false);
        setError(getErrorMessage(err));
        setScanResult(null);
      }
    },
    [startScanREST, useWebSocketMode]
  );

  const reset = useCallback(() => {
    setScanning(false);
    setScanResult(null);
    setScanProgress(null);
    setError(null);
    disconnect();
  }, [disconnect]);

  return {
    scanning,
    scanResult,
    scanProgress,
    error,
    startScan,
    reset,
    useWebSocketMode,
  };
}
