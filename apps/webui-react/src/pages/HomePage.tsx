import { useUIStore } from '../stores/uiStore';
import SearchInterface from '../components/SearchInterface';
import CollectionsDashboard from '../components/CollectionsDashboard';
import ActiveOperationsTab from '../components/ActiveOperationsTab';
import ErrorBoundary from '../components/ErrorBoundary';

function HomePage() {
  const activeTab = useUIStore((state) => state.activeTab);

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