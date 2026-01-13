import { Shield } from 'lucide-react';
import DatabaseSettings from './DatabaseSettings';

/**
 * AdminTab displays admin-only settings and operations.
 * This tab is only visible to superusers.
 *
 * Currently contains DatabaseSettings (with reset database functionality).
 * Phase 4 will add: Resource Limits, Performance, GPU & Memory, Search & Reranking.
 */
export default function AdminTab() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 flex items-center gap-2">
          <Shield className="h-5 w-5 text-amber-500" />
          Admin Settings
        </h3>
        <p className="mt-1 text-sm text-gray-500">
          Administrative operations and system configuration. These settings affect all users.
        </p>
      </div>

      <DatabaseSettings />
    </div>
  );
}
