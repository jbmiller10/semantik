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
  const routeControlledCollectionIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (collectionId) {
      if (activeTab !== 'collections') {
        hasSyncedRouteRef.current = false;
        setActiveTab('collections');
        return;
      }

      if (showCollectionDetailsModal !== collectionId) {
        hasSyncedRouteRef.current = false;
        setShowCollectionDetailsModal(collectionId);
      }

      routeControlledCollectionIdRef.current = collectionId;
      hasSyncedRouteRef.current = true;
      return;
    }

    const routeControlledCollectionId = routeControlledCollectionIdRef.current;

    if (routeControlledCollectionId !== null) {
      if (showCollectionDetailsModal === routeControlledCollectionId) {
        setShowCollectionDetailsModal(null);
      }

      routeControlledCollectionIdRef.current = null;
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
    if (!hasSyncedRouteRef.current || collectionId) {
      return;
    }

    if (activeTab !== 'collections') {
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
