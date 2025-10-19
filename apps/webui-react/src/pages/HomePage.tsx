import { useUIStore } from '../stores/uiStore';
import type { UIState } from '../stores/uiStore';
import { useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import SearchInterface from '../components/SearchInterface';
import CollectionsDashboard from '../components/CollectionsDashboard';
import ActiveOperationsTab from '../components/ActiveOperationsTab';
import ErrorBoundary from '../components/ErrorBoundary';

function HomePage() {
  const { collectionId } = useParams<{ collectionId?: string }>();
  const navigate = useNavigate();
  const activeTab = useUIStore((state: UIState) => state.activeTab);
  const setActiveTab = useUIStore((state: UIState) => state.setActiveTab);
  const setShowCollectionDetailsModal = useUIStore((state: UIState) => state.setShowCollectionDetailsModal);
  const showCollectionDetailsModal = useUIStore((state: UIState) => state.showCollectionDetailsModal);
  const hasSyncedRouteRef = useRef(false);

  useEffect(() => {
    if (collectionId) {
      if (activeTab !== 'collections') {
        setActiveTab('collections');
      }
      if (showCollectionDetailsModal !== collectionId) {
        setShowCollectionDetailsModal(collectionId);
      }
    } else if (showCollectionDetailsModal !== null) {
      setShowCollectionDetailsModal(null);
    }
    hasSyncedRouteRef.current = true;
  }, [
    collectionId,
    activeTab,
    showCollectionDetailsModal,
    setActiveTab,
    setShowCollectionDetailsModal,
  ]);

  useEffect(() => {
    if (!hasSyncedRouteRef.current) {
      return;
    }

    if (activeTab !== 'collections' && collectionId) {
      navigate('/', { replace: true });
    }
  }, [activeTab, collectionId, navigate]);

  return (
    <>
      {activeTab === 'search' && <SearchInterface />}
      {activeTab === 'collections' && (
        <ErrorBoundary>
          <CollectionsDashboard />
        </ErrorBoundary>
      )}
      {activeTab === 'operations' && (
        <ErrorBoundary>
          <ActiveOperationsTab />
        </ErrorBoundary>
      )}
    </>
  );
}

export default HomePage;
