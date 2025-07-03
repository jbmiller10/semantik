import { useState, useCallback, useRef } from 'react';
import { useWebSocket } from './useWebSocket';
import api from '../services/api';

interface ScanResult {
  files: string[];
  total_files: number;
  total_size: number;
  warnings?: Array<{
    type: string;
    message: string;
    severity: string;
  }>;
}

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
  const wsRef = useRef<WebSocket | null>(null);
  const scanIdRef = useRef<string>(scanId || Math.random().toString(36).substring(7));

  // WebSocket URL
  const wsUrl = scanning && useWebSocketMode 
    ? `ws://${window.location.host}/ws/scan/${scanIdRef.current}` 
    : null;

  const { disconnect } = useWebSocket(wsUrl, {
    onOpen: () => {
      console.log('Scan WebSocket connected');
    },
    onMessage: (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Scan WebSocket message:', data);

        switch (data.type) {
          case 'started':
            setScanProgress({ files_scanned: 0 });
            break;

          case 'counting':
            setScanProgress({
              current_path: 'Counting files...',
              files_scanned: 0,
            });
            break;

          case 'progress':
            setScanProgress({
              current_path: data.current_path,
              files_scanned: data.files_scanned,
              total_files: data.total_files,
              percentage: data.percentage,
            });
            break;

          case 'completed':
            setScanning(false);
            setScanResult({
              files: data.files.map((f: any) => typeof f === 'string' ? f : f.path),
              total_files: data.count || data.files.length,
              total_size: data.total_size || 0,
              warnings: data.warnings || [],
            });
            setScanProgress(null);
            break;
            
          case 'warning':
            // Handle warning messages during scan
            if (data.warning) {
              setScanResult(prev => prev ? {
                ...prev,
                warnings: [...(prev.warnings || []), data.warning]
              } : null);
            }
            break;

          case 'error':
            setScanning(false);
            setError(data.error || data.message || 'Scan failed');
            setScanProgress(null);
            break;

          case 'cancelled':
            setScanning(false);
            setScanProgress(null);
            break;
        }
      } catch (error) {
        console.error('Error parsing scan WebSocket message:', error);
      }
    },
    onError: () => {
      console.error('Scan WebSocket error, falling back to REST API');
      setUseWebSocketMode(false);
      // The startScan function will be called again automatically
    },
    onClose: () => {
      console.log('Scan WebSocket closed');
      setScanning(false);
    },
  });

  const startScanREST = useCallback(
    async (directory: string) => {
      setError(null);
      setScanResult(null);
      setScanProgress(null);
      setScanning(true);

      try {
        const response = await api.post('/api/scan-directory', {
          path: directory,
          recursive: true,
          scan_id: scanIdRef.current,
        });

        const { files, count, total_size, warnings } = response.data;
        
        // Calculate total size if not provided
        const totalSize = total_size || files.reduce((sum: number, file: any) => sum + file.size, 0);
        
        setScanResult({
          files: files.map((f: any) => f.path),
          total_files: count,
          total_size: totalSize,
          warnings: warnings || [],
        });
        setError(null);
      } catch (err: any) {
        console.error('Scan error:', err);
        const errorMessage = err.response?.data?.detail || err.message || 'Failed to scan directory';
        setError(errorMessage);
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

      // If WebSocket mode failed previously, use REST directly
      if (!useWebSocketMode) {
        return startScanREST(directory);
      }

      // Try WebSocket first
      setError(null);
      setScanResult(null);
      setScanProgress(null);
      setScanning(true);

      // Send scan request via REST API, which will trigger WebSocket updates
      try {
        const response = await api.post('/api/scan-directory', {
          path: directory,
          recursive: true,
          scan_id: scanIdRef.current,
          use_websocket: true,
        });

        // If REST returns immediately with results, WebSocket might not be available
        if (response.data.files) {
          setUseWebSocketMode(false);
          const { files, count } = response.data;
          const totalSize = files.reduce((sum: number, file: any) => sum + file.size, 0);
          
          setScanResult({
            files: files.map((f: any) => f.path),
            total_files: count,
            total_size: totalSize,
          });
          setScanning(false);
        }
      } catch (err: any) {
        console.error('Scan initialization error:', err);
        // Fall back to REST
        setUseWebSocketMode(false);
        return startScanREST(directory);
      }
    },
    [startScanREST, useWebSocketMode]
  );

  const reset = useCallback(() => {
    setScanning(false);
    setScanResult(null);
    setScanProgress(null);
    setError(null);
    if (wsRef.current) {
      disconnect();
    }
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