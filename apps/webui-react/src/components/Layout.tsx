import { Outlet, Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { useUIStore } from '../stores/uiStore';
import Toast from './Toast';
import DocumentViewerModal from './DocumentViewerModal';
import JobMetricsModal from './JobMetricsModal';
import CollectionDetailsModal from './CollectionDetailsModal';

function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();
  const { activeTab, setActiveTab } = useUIStore();
  
  // Check if we're on the settings page
  const isSettingsPage = location.pathname === '/settings';

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-semibold text-gray-900">Semantik</h1>
              <span className="ml-2 text-sm text-gray-500">Document Embedding Pipeline</span>
            </div>
            
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-700">{user?.username}</span>
              {import.meta.env.DEV && (
                <Link
                  to="/verification"
                  className="text-sm text-purple-600 hover:text-purple-900"
                >
                  Verification
                </Link>
              )}
              <Link
                to={isSettingsPage ? "/" : "/settings"}
                className="text-sm text-gray-600 hover:text-gray-900"
              >
                {isSettingsPage ? "← Back" : "Settings"}
              </Link>
              <button
                onClick={handleLogout}
                className="text-sm text-gray-600 hover:text-gray-900"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tabs - Only show on home page */}
        {!isSettingsPage && (
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab('collections')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'collections'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Collections
              </button>
              <button
                onClick={() => setActiveTab('operations')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'operations'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Active Operations
              </button>
              <button
                onClick={() => setActiveTab('search')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'search'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Search
              </button>
              <button
                onClick={() => setActiveTab('create')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'create'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Create Job
              </button>
              <button
                onClick={() => setActiveTab('jobs')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'jobs'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Jobs
              </button>
            </nav>
          </div>
        )}

        {/* Tab Content */}
        <div className="mt-8">
          <Outlet />
        </div>
      </main>

      {/* Toast Container */}
      <Toast />
      
      {/* Document Viewer Modal */}
      <DocumentViewerModal />
      
      {/* Job Metrics Modal */}
      <JobMetricsModal />
      
      {/* Collection Details Modal */}
      <CollectionDetailsModal />
    </div>
  );
}

export default Layout;