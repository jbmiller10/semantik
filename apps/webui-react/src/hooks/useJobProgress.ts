import { useEffect } from 'react';
import { useJobsStore } from '../stores/jobsStore';
import { useUIStore } from '../stores/uiStore';
import { useWebSocket } from './useWebSocket';

export function useJobProgress(jobId: string | null, enabled: boolean = true) {
  const { updateJob, setActiveJob } = useJobsStore();
  const addToast = useUIStore((state) => state.addToast);

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = jobId && enabled ? `${protocol}//${window.location.host}/ws/${jobId}` : null;

  const { disconnect } = useWebSocket(url, {
    autoReconnect: true,
    reconnectInterval: 5000,
    reconnectAttempts: 3,
    onOpen: () => {
      if (jobId) {
        setActiveJob(jobId, true);
      }
    },
    onMessage: (event) => {
      if (!jobId) return;

      try {
        const data = JSON.parse(event.data);
        switch (data.type) {
          case 'job_started':
            updateJob(jobId, {
              status: 'processing',
            });
            break;

          case 'file_processing':
            // Handle file processing updates
            const processedFiles = data.processed_files || 0;
            const totalFiles = data.total_files || 0;
            const progress = totalFiles > 0 ? Math.round((processedFiles / totalFiles) * 100) : 0;
            
            updateJob(jobId, {
              progress,
              processed_documents: processedFiles,
              total_documents: totalFiles,
              current_file: data.current_file,
            });
            break;

          case 'file_completed':
            // Handle file completion
            const completedFiles = data.processed_files || 0;
            const totalFilesCount = data.total_files || 0;
            const fileProgress = totalFilesCount > 0 ? Math.round((completedFiles / totalFilesCount) * 100) : 0;
            
            updateJob(jobId, {
              progress: fileProgress,
              processed_documents: completedFiles,
              total_documents: totalFilesCount,
            });
            break;

          case 'progress':
            // Legacy progress format support
            updateJob(jobId, {
              progress: data.progress,
              processed_documents: data.processed_documents || data.processed_files,
              total_documents: data.total_documents || data.total_files,
            });
            break;

          case 'metrics':
            updateJob(jobId, {
              metrics: {
                processing_rate: data.processing_rate || data.docs_per_second,
                estimated_time_remaining: data.estimated_time_remaining || data.eta_seconds,
                queue_position: data.queue_position,
              },
            });
            break;

          case 'job_completed':
          case 'completed':
            updateJob(jobId, {
              status: 'completed',
              progress: 100,
            });
            addToast({
              type: 'success',
              message: `Job ${data.job_name ? `"${data.job_name}"` : ''} completed successfully!`,
            });
            break;

          case 'job_cancelled':
          case 'cancelled':
            updateJob(jobId, {
              status: 'cancelled',
            });
            addToast({
              type: 'info',
              message: `Job ${data.job_name ? `"${data.job_name}"` : ''} was cancelled`,
            });
            break;

          case 'error':
            updateJob(jobId, {
              status: 'failed',
              error: data.error || data.message,
            });
            addToast({
              type: 'error',
              message: `Job failed: ${data.error || data.message}`,
            });
            break;

          default:
            console.warn('Unknown WebSocket message type:', data.type);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    },
    onError: () => {
      // Silently handle errors - WebSocket connections can fail for various reasons
    },
    onClose: () => {
      if (jobId) {
        setActiveJob(jobId, false);
      }
    },
  });

  useEffect(() => {
    return () => {
      if (jobId) {
        setActiveJob(jobId, false);
      }
    };
  }, [jobId, setActiveJob]);

  return { disconnect };
}