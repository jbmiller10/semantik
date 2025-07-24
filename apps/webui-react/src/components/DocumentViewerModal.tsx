import { useEffect } from 'react';
import { useUIStore } from '../stores/uiStore';
import { useSearchStore } from '../stores/searchStore';
import DocumentViewer from './DocumentViewer';

function DocumentViewerModal() {
  const { showDocumentViewer, setShowDocumentViewer } = useUIStore();
  const searchQuery = useSearchStore((state) => state.searchParams.query);

  useEffect(() => {
    // Prevent body scroll when modal is open
    if (showDocumentViewer) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }

    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [showDocumentViewer]);

  if (!showDocumentViewer) return null;

  const { collectionId, docId } = showDocumentViewer;

  return (
    <DocumentViewer
      collectionId={collectionId}
      docId={docId}
      query={searchQuery}
      onClose={() => setShowDocumentViewer(null)}
    />
  );
}

export default DocumentViewerModal;