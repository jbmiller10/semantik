import { useUIStore } from '../stores/uiStore';
import CreateJobForm from '../components/CreateJobForm';
import JobList from '../components/JobList';
import SearchInterface from '../components/SearchInterface';
import CollectionList from '../components/CollectionList';

function HomePage() {
  const activeTab = useUIStore((state) => state.activeTab);

  return (
    <>
      {activeTab === 'create' && <CreateJobForm />}
      {activeTab === 'jobs' && <JobList />}
      {activeTab === 'search' && <SearchInterface />}
      {activeTab === 'collections' && <CollectionList />}
    </>
  );
}

export default HomePage;