import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Settings,
  Shield,
  Server,
  Puzzle,
  Terminal,
  ArrowLeft,
} from 'lucide-react';
import { useAuthStore } from '../stores/authStore';
import PreferencesTab from '../components/settings/PreferencesTab';
import AdminTab from '../components/settings/AdminTab';
import SystemTab from '../components/settings/SystemTab';
import PluginsSettings from '../components/settings/PluginsSettings';
import MCPProfilesSettings from '../components/settings/MCPProfilesSettings';

type SettingsTab = 'preferences' | 'admin' | 'system' | 'plugins' | 'mcp';

interface TabConfig {
  id: SettingsTab;
  label: string;
  icon: typeof Settings;
  requiresSuperuser?: boolean;
}

const tabs: TabConfig[] = [
  { id: 'preferences', label: 'Preferences', icon: Settings },
  { id: 'admin', label: 'Admin', icon: Shield, requiresSuperuser: true },
  { id: 'system', label: 'System', icon: Server },
  { id: 'plugins', label: 'Plugins', icon: Puzzle },
  { id: 'mcp', label: 'MCP Profiles', icon: Terminal },
];

function SettingsPage() {
  const navigate = useNavigate();
  const user = useAuthStore((state) => state.user);
  const isSuperuser = user?.is_superuser ?? false;
  const [activeTab, setActiveTab] = useState<SettingsTab>('preferences');

  // Redirect non-superuser away from admin tab
  useEffect(() => {
    if (activeTab === 'admin' && !isSuperuser) {
      setActiveTab('preferences');
    }
  }, [activeTab, isSuperuser]);

  // Filter tabs based on user permissions
  const visibleTabs = tabs.filter(
    (tab) => !tab.requiresSuperuser || isSuperuser
  );

  return (
    <div className="space-y-6">
      {/* Page Header with Back Button */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Settings</h2>
          <p className="mt-1 text-sm text-gray-500">
            Manage your preferences, plugins, and system settings
          </p>
        </div>
        <button
          onClick={() => navigate('/')}
          className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          <ArrowLeft className="mr-2 -ml-0.5 h-4 w-4" />
          Back to Home
        </button>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8" aria-label="Settings tabs">
          {visibleTabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm
                  ${
                    isActive
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }
                `}
              >
                <Icon
                  className={`inline-block w-5 h-5 mr-2 -mt-0.5 ${
                    isActive ? 'text-blue-500' : 'text-gray-400'
                  }`}
                />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="mt-6">
        {activeTab === 'preferences' && <PreferencesTab />}
        {activeTab === 'admin' && isSuperuser && <AdminTab />}
        {activeTab === 'system' && <SystemTab />}
        {activeTab === 'plugins' && <PluginsSettings />}
        {activeTab === 'mcp' && <MCPProfilesSettings />}
      </div>
    </div>
  );
}

export default SettingsPage;
