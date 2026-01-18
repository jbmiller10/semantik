import { useEffect } from 'react';
import { Outlet, Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { useUIStore } from '../stores/uiStore';
import { useOperationsSocket } from '../hooks/useOperationsSocket';
import Toast from './Toast';
import DocumentViewerModal from './DocumentViewerModal';
import CollectionDetailsModal from './CollectionDetailsModal';
import ThemeToggle from './ThemeToggle';
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
      {/* Header */}
      <header className="sticky top-0 z-50 bg-[var(--bg-secondary)] border-b border-[var(--border)] shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="h-8 w-8 bg-ink-900 dark:bg-paper-100 rounded-lg flex items-center justify-center">
                <span className="text-paper-100 dark:text-ink-900 font-serif font-bold text-xl">S</span>
              </div>
              <div>
                <h1 className="text-xl font-serif font-semibold text-[var(--text-primary)] tracking-tight">Semantik</h1>
                <p className="text-[10px] uppercase tracking-widest text-[var(--text-muted)] font-medium -mt-0.5">Semantic Search</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="hidden sm:flex items-center space-x-2 text-sm">
                <span className="text-[var(--text-muted)]">Signed in as</span>
                <span className="text-[var(--text-primary)] font-medium">{user?.username}</span>
              </div>

              <div className="h-4 w-px bg-[var(--border)] hidden sm:block"></div>

              <div className="flex items-center space-x-2">
                <ThemeToggle />

                {import.meta.env.DEV && (
                  <Link
                    to="/verification"
                    className="text-sm font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors px-2 py-1"
                  >
                    Verify
                  </Link>
                )}
                <Link
                  to={isSettingsPage ? "/" : "/settings"}
                  className="text-sm font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors px-2 py-1"
                >
                  {isSettingsPage ? "Back" : "Settings"}
                </Link>
                <button
                  onClick={handleLogout}
                  className="btn-secondary text-xs px-3 py-1.5"
                >
                  Sign out
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Navigation - Only show on home page */}
        {!isSettingsPage && (
          <div className="mb-8">
            <nav className="tab-nav">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`tab-item ${activeTab === tab.id ? 'tab-item-active' : ''}`}
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
