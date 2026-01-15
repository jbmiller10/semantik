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

  type TabId = 'search' | 'collections' | 'operations';
  const tabs: { id: TabId; label: string }[] = [
    { id: 'collections', label: 'Collections' },
    { id: 'operations', label: 'Active Operations' },
    { id: 'search', label: 'Search' },
  ];

  return (
    <div className="min-h-screen">
      {/* Glass Header */}
      <header className="sticky top-0 z-50 glass-panel border-x-0 border-t-0 border-b border-white/5 rounded-none backdrop-blur-xl bg-void-950/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="h-8 w-8 bg-signal-600 rounded-lg shadow-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">S</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-white tracking-tight">SEMANTIK</h1>
                <p className="text-[10px] uppercase tracking-widest text-gray-500 font-semibold -mt-1 pl-0.5">Intelligence Pipeline</p>
              </div>
            </div>

            <div className="flex items-center space-x-6">
              <div className="hidden sm:flex items-center space-x-2 text-sm font-medium">
                <span className="text-gray-500">Operator:</span>
                <span className="text-gray-200">{user?.username}</span>
              </div>

              <div className="h-4 w-px bg-white/10 hidden sm:block"></div>

              <div className="flex items-center space-x-4">
                {import.meta.env.DEV && (
                  <Link
                    to="/verification"
                    className="text-sm font-medium text-signal-400 hover:text-signal-300 transition-colors"
                  >
                    Verification
                  </Link>
                )}
                <Link
                  to={isSettingsPage ? "/" : "/settings"}
                  className="text-sm font-medium text-gray-400 hover:text-white transition-colors"
                >
                  {isSettingsPage ? "← Back" : "Settings"}
                </Link>
                <button
                  onClick={handleLogout}
                  className="px-4 py-1.5 text-xs font-bold uppercase tracking-wider text-white bg-void-800 hover:bg-void-700 border border-white/10 rounded-md transition-all duration-200"
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
            <nav className="flex space-x-1 p-1 bg-void-900/50 backdrop-blur-sm rounded-lg border border-white/5 w-fit">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    px-5 py-2 text-sm font-medium rounded-md transition-all duration-200
                    ${activeTab === tab.id
                      ? 'bg-signal-600 text-white shadow-sm'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
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
