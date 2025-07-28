import { useEffect, useRef } from 'react';
import { useSearchStore } from '../stores/searchStore';
import { systemApi } from '../services/api/v2/system';

/**
 * Hook to check and monitor reranking availability
 * 
 * This hook will:
 * 1. Check system status on mount
 * 2. Monitor search errors for GPU-related issues
 * 3. Update the search store accordingly
 */
export function useRerankingAvailability() {
  const { 
    setRerankingAvailable, 
    setRerankingModelsLoading,
    error,
    searchParams
  } = useSearchStore();
  
  // Track if we've already detected unavailability
  const unavailabilityDetected = useRef(false);

  useEffect(() => {
    const checkAvailability = async () => {
      // Don't check if we already know it's unavailable
      if (unavailabilityDetected.current) {
        return;
      }
      
      setRerankingModelsLoading(true);
      
      try {
        const status = await systemApi.getStatus();
        setRerankingAvailable(status.reranking_available);
        if (!status.reranking_available) {
          unavailabilityDetected.current = true;
        }
      } catch (error: any) {
        console.warn('System status endpoint not available, assuming reranking is available');
        // If it's a 404, the endpoint doesn't exist yet - assume available
        if (error?.response?.status === 404) {
          setRerankingAvailable(true);
        } else {
          // For other errors, be conservative
          console.error('Failed to check reranking availability:', error);
          setRerankingAvailable(true);
        }
      } finally {
        setRerankingModelsLoading(false);
      }
    };

    checkAvailability();
  }, [setRerankingAvailable, setRerankingModelsLoading]);

  // Monitor search errors for GPU-related issues
  useEffect(() => {
    if (error && searchParams.useReranker) {
      // Check if error indicates GPU/reranking unavailability
      const errorLower = error.toLowerCase();
      if (
        errorLower.includes('gpu') ||
        errorLower.includes('cuda') ||
        errorLower.includes('rerank') ||
        errorLower.includes('model not found') ||
        errorLower.includes('out of memory')
      ) {
        setRerankingAvailable(false);
      }
    }
  }, [error, searchParams.useReranker, setRerankingAvailable]);
}