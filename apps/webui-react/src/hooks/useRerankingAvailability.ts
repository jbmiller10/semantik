import { useEffect } from 'react';
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

  useEffect(() => {
    const checkAvailability = async () => {
      setRerankingModelsLoading(true);
      
      try {
        const status = await systemApi.getStatus();
        setRerankingAvailable(status.reranking_available);
      } catch (error) {
        console.error('Failed to check reranking availability:', error);
        // Default to available unless we know otherwise
        setRerankingAvailable(true);
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