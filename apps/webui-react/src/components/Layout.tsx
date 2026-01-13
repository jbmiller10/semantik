import { useEffect } from 'react';
import { Outlet, Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { useUIStore } from '../stores/uiStore';
import { useOperationsSocket } from '../hooks/useOperationsSocket';
import Toast from './Toast';
import DocumentViewerModal from './DocumentViewerModal';
import CollectionDetailsModal from './CollectionDetailsModal';
import { registerNavigationHandler } from '../services/navigation';

function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();
  const { activeTab, setActiveTab } = useUIStore();

  // Global WebSocket subscription for operation updates
  // Ensures collection stats update in real-time across all views
  useOperationsSocket();

  useEffect(() => {
    registerNavigationHandler(navigate);
  }, [navigate]);

  // Check if we're on the settings page
  const isSettingsPage = location.pathname === '/settings';

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  const tabs = [
    { id: 'collections', label: 'Collections' },
    { id: 'operations', label: 'Active Operations' },
    { id: 'search', label: 'Search' },
  ];

  return (
    <div className="min-h-screen">
      {/* Glass Header */}
      <header className="sticky top-0 z-50 glass-panel border-b-0 rounded-b-xl mx-4 mt-2">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="h-8 w-8 bg-gradient-to-br from-brand-500 to-accent-500 rounded-lg shadow-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">S</span>
              </div>
              <div>
                <h1 className="text-xl font-bold heading-gradient">Semantik</h1>
                <p className="text-xs text-brand-600/70 font-medium -mt-1">Document Pipeline</p>
              </div>
            </div>

            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2 text-sm font-medium">
                <span className="text-brand-900/50">Welcome,</span>
                <span className="text-brand-900">{user?.username}</span>
              </div>

              <div className="h-4 w-px bg-brand-200"></div>

              <div className="flex items-center space-x-4">
                {import.meta.env.DEV && (
                  <Link
                    to="/verification"
                    className="text-sm font-medium text-purple-600 hover:text-purple-700 transition-colors"
                  >
                    Verification
                  </Link>
                )}
                <Link
                  to={isSettingsPage ? "/" : "/settings"}
                  className="text-sm font-medium text-brand-600 hover:text-brand-800 transition-colors"
                >
                  {isSettingsPage ? "← Back" : "Settings"}
                </Link>
                <button
                  onClick={handleLogout}
                  className="px-4 py-1.5 text-sm font-medium text-white bg-brand-600 hover:bg-brand-700 rounded-lg shadow-md hover:shadow-lg transition-all duration-200"
                >
                  Logout
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Modern Tabs - Only show on home page */}
        {!isSettingsPage && (
          <div className="mb-8">
            <nav className="flex space-x-1 p-1 bg-white/50 backdrop-blur-sm rounded-xl border border-white/40 w-fit">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`
                    px-6 py-2.5 text-sm font-medium rounded-lg transition-all duration-200
                    ${activeTab === tab.id
                      ? 'bg-white text-brand-600 shadow-sm ring-1 ring-black/5'
                      : 'text-brand-600/70 hover:text-brand-800 hover:bg-white/30'
                    }
                  `}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
        )}

        {/* Tab Content */}
        <div className="animate-fade-in">
          <Outlet />
        </div>
      </main>

      {/* Toast Container */}
      <Toast />

      {/* Document Viewer Modal */}
      <DocumentViewerModal />

      {/* Collection Details Modal */}
      <CollectionDetailsModal />
    </div>
  );
}

export default Layout;
