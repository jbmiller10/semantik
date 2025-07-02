import { useState, useCallback } from 'react';
import api from '../services/api';

interface ScanResult {
  files: string[];
  total_files: number;
  total_size: number;
}

export function useDirectoryScan(_scanId?: string) {
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);

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
        const response = await api.post('/api/scan-directory', {
          path: directory,
          recursive: true
        });

        const { files, count } = response.data;
        
        // Calculate total size
        const totalSize = files.reduce((sum: number, file: any) => sum + file.size, 0);
        
        setScanResult({
          files: files.map((f: any) => f.path),
          total_files: count,
          total_size: totalSize,
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