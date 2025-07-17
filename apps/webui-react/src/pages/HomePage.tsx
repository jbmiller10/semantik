import { useUIStore } from '../stores/uiStore';
import CreateJobForm from '../components/CreateJobForm';
import JobList from '../components/JobList';
import SearchInterface from '../components/SearchInterface';
import CollectionsDashboard from '../components/CollectionsDashboard';
import ErrorBoundary from '../components/ErrorBoundary';

function HomePage() {
  const activeTab = useUIStore((state) => state.activeTab);

  return (
    <>
      {activeTab === 'create' && <CreateJobForm />}
      {activeTab === 'jobs' && <JobList />}
      {activeTab === 'search' && <SearchInterface />}
      {activeTab === 'collections' && (
        <ErrorBoundary>
          <CollectionsDashboard />
        </ErrorBoundary>
      )}
    </>
  );
}

export default HomePage;