import { useState } from 'react';
import { useMCPProfiles } from '../../hooks/useMCPProfiles';
import type { MCPProfile } from '../../types/mcp-profile';
import ProfileCard from '../mcp/ProfileCard';
import ProfileFormModal from '../mcp/ProfileFormModal';
import ConfigModal from '../mcp/ConfigModal';
import DeleteConfirmModal from '../mcp/DeleteConfirmModal';

export default function MCPProfilesSettings() {
  const { data: profiles, isLoading, error, refetch } = useMCPProfiles();

  // Modal state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [editingProfile, setEditingProfile] = useState<MCPProfile | null>(null);
  const [configProfile, setConfigProfile] = useState<MCPProfile | null>(null);
  const [deletingProfile, setDeletingProfile] = useState<MCPProfile | null>(null);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <svg
          className="animate-spin h-8 w-8 text-[var(--text-muted)]"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
        <span className="ml-3 text-[var(--text-secondary)]">Loading MCP profiles...</span>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex">
          <svg
            className="h-5 w-5 text-red-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">
              Error loading MCP profiles
            </h3>
            <p className="mt-1 text-sm text-red-700">
              {error instanceof Error ? error.message : 'Unknown error occurred'}
            </p>
            <button
              onClick={() => refetch()}
              className="mt-2 text-sm font-medium text-red-600 hover:text-red-500"
            >
              Try again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-medium text-[var(--text-primary)]">MCP Profiles</h3>
          <p className="mt-1 text-sm text-[var(--text-secondary)]">
            Configure search profiles for MCP clients like Claude Desktop
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-[var(--accent-primary)] border border-transparent rounded-md hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-[var(--accent-primary)] focus:ring-offset-2"
        >
          <svg
            className="w-4 h-4 mr-2"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 4v16m8-8H4"
            />
          </svg>
          Create Profile
        </button>
      </div>

      {/* Info Box */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex">
          <svg
            className="h-5 w-5 text-blue-400 flex-shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800">
              What are MCP Profiles?
            </h3>
            <p className="mt-1 text-sm text-blue-700">
              MCP (Model Context Protocol) profiles let you expose your Semantik
              collections to AI assistants like Claude Desktop. Each profile
              creates a search tool that the AI can use to search your selected
              collections with preconfigured settings.
            </p>
          </div>
        </div>
      </div>

      {/* Empty State */}
      {(!profiles || profiles.length === 0) && (
        <div className="text-center py-12 bg-[var(--bg-tertiary)] rounded-lg border-2 border-dashed border-[var(--border)]">
          <svg
            className="mx-auto h-12 w-12 text-[var(--text-muted)]"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
          <h3 className="mt-2 text-sm font-medium text-[var(--text-primary)]">
            No MCP profiles
          </h3>
          <p className="mt-1 text-sm text-[var(--text-secondary)]">
            Get started by creating a new profile to expose your collections to
            AI assistants.
          </p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="mt-4 inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-[var(--accent-primary)] border border-transparent rounded-md hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-[var(--accent-primary)] focus:ring-offset-2"
          >
            <svg
              className="w-4 h-4 mr-2"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 4v16m8-8H4"
              />
            </svg>
            Create Profile
          </button>
        </div>
      )}

      {/* Profile List */}
      {profiles && profiles.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {profiles.map((profile) => (
            <ProfileCard
              key={profile.id}
              profile={profile}
              onEdit={() => setEditingProfile(profile)}
              onDelete={() => setDeletingProfile(profile)}
              onViewConfig={() => setConfigProfile(profile)}
            />
          ))}
        </div>
      )}

      {/* Modals */}
      {showCreateModal && (
        <ProfileFormModal onClose={() => setShowCreateModal(false)} />
      )}

      {editingProfile && (
        <ProfileFormModal
          profile={editingProfile}
          onClose={() => setEditingProfile(null)}
        />
      )}

      {configProfile && (
        <ConfigModal
          profile={configProfile}
          onClose={() => setConfigProfile(null)}
        />
      )}

      {deletingProfile && (
        <DeleteConfirmModal
          profile={deletingProfile}
          onClose={() => setDeletingProfile(null)}
        />
      )}
    </div>
  );
}
